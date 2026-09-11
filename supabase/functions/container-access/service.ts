export type Fetch = (request: Request) => Promise<Response>;
type ObjectValue = Record<string, unknown>;
const lifetime = 15 * 60 * 1000;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const connectionId = /^[a-z][a-z0-9_-]{0,63}$/;
const encoder = new TextEncoder();

class Failure extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
function object(value: unknown): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Failure(400, "Invalid object");
  return Object.fromEntries(Object.entries(value));
}
function string(value: unknown): string {
  if (typeof value !== "string" || !value || value.length > 16384) throw new Failure(400, "Invalid string");
  return value;
}
function date(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const text = string(value);
  if (!Number.isFinite(Date.parse(text))) throw new Failure(400, "Invalid expiry");
  return new Date(text).toISOString();
}
function base64(bytes: Uint8Array): string { return btoa(String.fromCharCode(...bytes)); }
function unbase64(text: string): Uint8Array { return Uint8Array.from(atob(text), (c) => c.charCodeAt(0)); }
export async function hash(token: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(token));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export interface Configuration {
  url: string;
  serviceKey: string;
  encryptionKey: string; // base64-encoded 32-byte AES key, held only by this backend
  fetch?: Fetch;
  now?: () => number;
}

export function createHandler(config: Configuration): Fetch {
  const send = config.fetch ?? fetch;
  const now = config.now ?? Date.now;
  const dbURL = config.url.replace(/\/$/, "");
  const rawKey = unbase64(config.encryptionKey);
  if (rawKey.length !== 32) throw new Error("IVON_CREDENTIAL_KEY must encode 32 bytes");
  const key = crypto.subtle.importKey("raw", rawKey, "AES-GCM", false, ["encrypt", "decrypt"]);

  async function db(table: string, params: Record<string, string>, method = "GET", body?: unknown): Promise<unknown> {
    const url = new URL(`${dbURL}/rest/v1/${table}`);
    url.search = new URLSearchParams(params).toString();
    const response = await send(new Request(url, {
      method, redirect: "error", signal: AbortSignal.timeout(10000),
      headers: { apikey: config.serviceKey, Authorization: `Bearer ${config.serviceKey}`,
        "Content-Type": "application/json", Prefer: "return=representation" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }));
    if (!response.ok) throw new Failure(503, "Credential storage unavailable");
    return response.status === 204 ? null : response.json();
  }
  async function one(table: string, params: Record<string, string>, method = "GET", body?: unknown): Promise<ObjectValue> {
    const rows = await db(table, params, method, body);
    if (!Array.isArray(rows) || rows.length !== 1) throw new Failure(404, "Resource unavailable");
    return object(rows[0]);
  }
  async function user(token: string): Promise<string> {
    // Existing CLI API tokens are an application credential; container grants never enter this path.
    if (token.startsWith("ivon_") && !token.startsWith("ivon_container_")) {
      const row = await one("ivon_users", { api_token: `eq.${token}`, select: "id" });
      return string(row.id);
    }
    const response = await send(new Request(`${dbURL}/auth/v1/user`, {
      redirect: "error", signal: AbortSignal.timeout(10000),
      headers: { apikey: config.serviceKey, Authorization: `Bearer ${token}` },
    }));
    if (!response.ok) throw new Failure(401, "Authentication required");
    const verified = object(await response.json());
    const id = string(verified.id);
    if (!uuid.test(id)) throw new Failure(401, "Authentication required");
    await one("ivon_users", { id: `eq.${id}`, select: "id" });
    return id;
  }
  async function grant(token: string, session: string): Promise<ObjectValue> {
    if (!token.startsWith("ivon_container_") || !uuid.test(session)) throw new Failure(401, "Invalid container grant");
    try {
      return await one("ivon_container_grants", {
        session_id: `eq.${session}`, token_hash: `eq.${await hash(token)}`,
        revoked_at: "is.null", expires_at: `gt.${new Date(now()).toISOString()}`,
        select: "session_id,user_id,expires_at",
      });
    } catch (error) {
      if (error instanceof Failure && error.status === 404) throw new Failure(401, "Container grant expired or revoked");
      throw error;
    }
  }
  async function issue(owner: string, session?: string, resume = false): Promise<ObjectValue> {
    const token = `ivon_container_${base64(crypto.getRandomValues(new Uint8Array(32))).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "")}`;
    const id = session ?? crypto.randomUUID();
    const values = { revoked_at: null, token_hash: await hash(token), expires_at: new Date(now() + lifetime).toISOString() };
    if (session) {
      await one("ivon_container_grants", { session_id: `eq.${id}`, user_id: `eq.${owner}`, ...(resume ? {} : { revoked_at: "is.null" }) }, "PATCH", values);
    } else {
      await one("ivon_container_grants", {}, "POST", { session_id: id, user_id: owner, ...values });
    }
    return { session_id: id, user_id: owner, token, expires_at: values.expires_at };
  }
  async function encrypt(secret: string, owner: string, id: string): Promise<ObjectValue> {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv,
      additionalData: encoder.encode(`${owner}:${id}`) }, await key, encoder.encode(secret));
    return { iv: base64(iv), ciphertext: base64(new Uint8Array(ciphertext)) };
  }
  async function decrypt(encrypted: unknown, owner: string, id: string): Promise<string> {
    try {
      const value = object(encrypted);
      const plaintext = await crypto.subtle.decrypt({ name: "AES-GCM", iv: unbase64(string(value.iv)),
        additionalData: encoder.encode(`${owner}:${id}`) }, await key, unbase64(string(value.ciphertext)));
      return new TextDecoder().decode(plaintext);
    } catch { throw new Failure(503, "Credential unavailable"); }
  }
  function metadata(id: string, value: ObjectValue): ObjectValue {
    return { id, provider: string(value.provider), account_label: string(value.account_label), expires_at: date(value.expires_at) };
  }

  return async (request: Request): Promise<Response> => {
    const headers = { "Cache-Control": "no-store", "Content-Type": "application/json" };
    try {
      const path = new URL(request.url).pathname.split("/container-access").pop() ?? "";
      const token = request.headers.get("Authorization")?.match(/^Bearer ([A-Za-z0-9_.-]{1,4096})$/)?.[1];
      if (!token) throw new Failure(401, "Authentication required");
      const method = request.method;
      const session = request.headers.get("X-Session-ID") ?? "";
      const isContainer = token.startsWith("ivon_container_");
      let result: unknown;
      if (path === "/sessions" && method === "POST" && !isContainer) {
        result = await issue(await user(token));
      } else if (/^\/sessions\/[0-9a-f-]+\/(renew|resume)$/i.test(path) && method === "POST" && !isContainer) {
        const id = path.split("/")[2];
        if (!uuid.test(id)) throw new Failure(400, "Invalid session ID");
        result = await issue(await user(token), id, path.endsWith("/resume"));
      } else if (/^\/sessions\/[0-9a-f-]+$/i.test(path) && method === "DELETE") {
        const id = path.split("/")[2];
        if (!uuid.test(id)) throw new Failure(400, "Invalid session ID");
        const owner = isContainer ? string((await grant(token, session)).user_id) : await user(token);
        if (isContainer && id !== session) throw new Failure(403, "Session mismatch");
        await one("ivon_container_grants", { session_id: `eq.${id}`, user_id: `eq.${owner}` }, "PATCH", { revoked_at: new Date(now()).toISOString() });
        result = { ok: true };
      } else {
        const access = isContainer ? await grant(token, session) : null;
        const owner = access ? string(access.user_id) : await user(token);
        if (path === "/settings" && method === "GET") {
          const row = await one("ivon_users", { id: `eq.${owner}`, select: "container_settings" });
          result = { user_id: owner, expires_at: access?.expires_at ?? null, settings: object(row.container_settings) };
        } else if (path === "/settings" && method === "PUT" && !isContainer) {
          const settings = object(await request.json());
          for (const [name, value] of Object.entries(settings)) {
            if (!["timezone", "provider", "model", "web_search_provider"].includes(name) || typeof value !== "string" || value.length > 256) throw new Failure(400, "Invalid setting");
          }
          await one("ivon_users", { id: `eq.${owner}` }, "PATCH", { container_settings: settings });
          result = { ok: true };
        } else if (path === "/connections" && method === "GET") {
          const row = await one("ivon_users", { id: `eq.${owner}`, select: "container_connections,facebook_user_name,facebook_token_expires_at,facebook_connected_at" });
          const connections = Object.entries(object(row.container_connections)).map(([id, value]) => metadata(id, object(value)));
          if (row.facebook_connected_at) connections.push({ id: "facebook", provider: "facebook", account_label: row.facebook_user_name ?? "Facebook", expires_at: date(row.facebook_token_expires_at) });
          result = { connections };
        } else if (path.startsWith("/connections/")) {
          const parts = path.split("/");
          const id = parts[2];
          if (!connectionId.test(id)) throw new Failure(400, "Invalid connection ID");
          if (parts.length === 4 && parts[3] === "credential" && method === "GET" && isContainer) {
            const row = await one("ivon_users", { id: `eq.${owner}`, select: id === "facebook" ? "facebook_access_token,facebook_token_expires_at" : "container_connections" });
            const value = id === "facebook" ? { expires_at: row.facebook_token_expires_at } : object(object(row.container_connections)[id] ?? {});
            const expires = date(value.expires_at);
            if (expires && Date.parse(expires) <= now()) throw new Failure(409, "Provider credential expired");
            if (id !== "facebook" && !value.secret) throw new Failure(404, "Credential unavailable");
            if (id === "facebook" && !row.facebook_access_token) throw new Failure(404, "Credential unavailable");
            const secret = id === "facebook" ? string(row.facebook_access_token) : await decrypt(value.secret, owner, id);
            // Recheck after storage/decryption so a concurrent revoke or rotation wins.
            await grant(token, session);
            result = { credential: secret, expires_at: access?.expires_at, credential_expires_at: expires };
          } else if (parts.length === 3 && !isContainer && (method === "PUT" || method === "DELETE") && id !== "facebook") {
            let value: unknown = null;
            if (method === "PUT") {
              const body = object(await request.json());
              value = { provider: string(body.provider), account_label: string(body.account_label), expires_at: date(body.expires_at), secret: await encrypt(string(body.credential), owner, id) };
            }
            await db("rpc/ivon_set_container_connection", {}, "POST", { owner_id: owner, connection_id: id, connection: value });
            result = { ok: true };
          } else throw new Failure(403, "Operation unavailable");
        } else throw new Failure(404, "Route unavailable");
      }
      return new Response(JSON.stringify(result), { headers });
    } catch (error) {
      const failure = error instanceof Failure ? error : new Failure(503, "Credential service unavailable");
      return new Response(JSON.stringify({ error: failure.message }), { status: failure.status, headers });
    }
  };
}

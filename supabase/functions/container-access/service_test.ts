import { createHandler, hash } from "./service.ts";

function assert(value: unknown, message = "assertion failed"): asserts value {
  if (!value) throw new Error(message);
}
type Row = Record<string, unknown>;
const alice = "11111111-1111-4111-8111-111111111111";
const bob = "22222222-2222-4222-8222-222222222222";

// Contract fixture for the external Auth and PostgREST HTTP APIs. The production handler is unchanged.
function setup() {
  const users: Row[] = [alice, bob].map((id) => ({ id, api_token: `ivon_${id}`, container_settings: {}, container_connections: {}, facebook_access_token: null }));
  const grants: Row[] = [];
  let time = Date.now();
  const calls: Request[] = [];
  const handler = createHandler({ url: "https://ops.example", serviceKey: "server-key", encryptionKey: btoa("k".repeat(32)), now: () => time,
    fetch: async (request) => {
      calls.push(request);
      const url = new URL(request.url);
      assert(request.headers.get("apikey") === "server-key");
      if (url.pathname === "/auth/v1/user") {
        const id = request.headers.get("Authorization")?.replace("Bearer jwt-", "");
        return Response.json({ id }, { status: [alice, bob].includes(id ?? "") ? 200 : 401 });
      }
      assert(request.headers.get("Authorization") === "Bearer server-key");
      const body = request.method === "GET" ? null : await request.json();
      if (url.pathname === "/rest/v1/rpc/ivon_set_container_connection") {
        const row = users.find((row) => row.id === body.owner_id)!;
        const value = Object.fromEntries(Object.entries(row.container_connections ?? {}));
        if (body.connection === null) delete value[body.connection_id];
        else value[body.connection_id] = body.connection;
        row.container_connections = value;
        return Response.json(null);
      }
      const table = url.pathname === "/rest/v1/ivon_users" ? users : grants;
      if (request.method === "POST") {
        table.push({ revoked_at: null, ...body });
        return Response.json([body]);
      }
      const selected = table.filter((row) => [...url.searchParams].every(([key, filter]) => {
        if (key === "select") return true;
        if (filter.startsWith("eq.")) return row[key] === filter.slice(3);
        if (filter === "is.null") return row[key] == null;
        if (filter.startsWith("gt.")) return String(row[key]) > filter.slice(3);
        throw new Error("Unsupported PostgREST filter");
      }));
      if (request.method === "PATCH") selected.forEach((row) => Object.assign(row, body));
      const fields = url.searchParams.get("select")?.split(",");
      return Response.json(selected.map((row) => fields ? Object.fromEntries(fields.map((field) => [field, row[field]])) : row));
    } });
  async function request(path: string, token: string, method = "GET", body?: unknown, session = "") {
    return handler(new Request("https://ops.example/functions/v1/container-access" + path, { method,
      headers: { Authorization: "Bearer " + token, "X-Session-ID": session }, body: body === undefined ? undefined : JSON.stringify(body) }));
  }
  async function issue(id = alice) {
    const result = await request("/sessions", "jwt-" + id, "POST");
    assert(result.status === 200);
    return result.json();
  }
  return { users, grants, calls, request, issue, advance: () => { time += 16 * 60 * 1000; } };
}

Deno.test("issue hashes tokens and binds owner from authentication", async () => {
  const test = setup();
  const grant = await test.issue();
  assert(grant.user_id === alice);
  assert(test.grants[0].token_hash === await hash(grant.token));
  assert(!JSON.stringify(test.grants).includes(grant.token));
  const unauthenticated = await test.request("/sessions", "bad", "POST", { user_id: bob });
  assert(unauthenticated.status === 401);
  assert((await test.request("/sessions", grant.token, "POST")).status !== 200);
});

Deno.test("credentials are encrypted, owner-bound, and fetched by one connection ID", async () => {
  const test = setup();
  const result = await test.request("/connections/openai", "jwt-" + alice, "PUT", { provider: "openai", account_label: "Personal", credential: "provider-secret" });
  assert(result.status === 200);
  assert(!JSON.stringify(test.users).includes("provider-secret"));
  const grant = await test.issue();
  const fetched = await test.request("/connections/openai/credential", grant.token, "GET", undefined, grant.session_id);
  assert(fetched.status === 200);
  assert((await fetched.json()).credential === "provider-secret");
  const other = await test.issue(bob);
  assert((await test.request("/connections/openai/credential", other.token, "GET", undefined, other.session_id)).status === 404);
  assert((await test.request("/connections/openai/credential", grant.token, "GET", undefined, other.session_id)).status === 401);
  assert((await test.request("/connections/openai", grant.token, "PUT", {}, grant.session_id)).status === 403);
  const listed = await test.request("/connections", grant.token, "GET", undefined, grant.session_id);
  const text = await listed.text();
  assert(!text.includes("ciphertext") && !text.includes("provider-secret"));
  assert(listed.headers.get("Cache-Control") === "no-store");
});

Deno.test("expiry, renewal, revocation, and independent sessions", async () => {
  const test = setup();
  const first = await test.issue();
  const second = await test.issue();
  assert((await test.request(`/sessions/${first.session_id}/renew`, "jwt-" + bob, "POST")).status === 404);
  const renewed = await (await test.request(`/sessions/${first.session_id}/renew`, "jwt-" + alice, "POST")).json();
  assert((await test.request("/settings", first.token, "GET", undefined, first.session_id)).status === 401);
  assert((await test.request("/settings", renewed.token, "GET", undefined, renewed.session_id)).status === 200);
  assert((await test.request(`/sessions/${second.session_id}`, renewed.token, "DELETE", undefined, first.session_id)).status === 403);
  assert((await test.request(`/sessions/${first.session_id}`, renewed.token, "DELETE", undefined, first.session_id)).status === 200);
  assert((await test.request(`/sessions/${first.session_id}/renew`, "jwt-" + alice, "POST")).status === 404);
  assert((await test.request(`/sessions/${first.session_id}/resume`, "jwt-" + alice, "POST")).status === 200);
  assert((await test.request("/settings", second.token, "GET", undefined, second.session_id)).status === 200);
  test.advance();
  assert((await test.request("/settings", second.token, "GET", undefined, second.session_id)).status === 401);
});

Deno.test("existing Facebook connection and expired provider credentials", async () => {
  const test = setup();
  test.users[0].facebook_access_token = "existing-facebook-secret";
  test.users[0].facebook_token_expires_at = "2000-01-01T00:00:00Z";
  const grant = await test.issue();
  assert((await test.request("/connections/facebook/credential", grant.token, "GET", undefined, grant.session_id)).status === 409);
  test.users[0].facebook_token_expires_at = null;
  const result = await test.request("/connections/facebook/credential", grant.token, "GET", undefined, grant.session_id);
  assert((await result.json()).credential === "existing-facebook-secret");
  test.users[0].facebook_access_token = null;
  assert((await test.request("/connections/facebook/credential", grant.token, "GET", undefined, grant.session_id)).status === 404);
});

Deno.test("moving ciphertext between users fails authentication", async () => {
  const test = setup();
  await test.request("/connections/openai", "jwt-" + alice, "PUT", { provider: "openai", account_label: "Personal", credential: "provider-secret" });
  test.users[1].container_connections = structuredClone(test.users[0].container_connections);
  const grant = await test.issue(bob);
  const result = await test.request("/connections/openai/credential", grant.token, "GET", undefined, grant.session_id);
  assert(result.status === 503);
  assert(!(await result.text()).includes("provider-secret"));
});

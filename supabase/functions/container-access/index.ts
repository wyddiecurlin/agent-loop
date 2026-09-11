import { createHandler } from "./service.ts";

function required(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`${name} is required`);
  return value;
}
Deno.serve(createHandler({
  url: required("SUPABASE_URL"),
  serviceKey: required("SUPABASE_SERVICE_ROLE_KEY"),
  encryptionKey: required("IVON_CREDENTIAL_KEY"),
}));

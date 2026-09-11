-- Target: ops (ftlvnvrzsukhsjgeqebl). Existing ivon_users.id references auth.users.id.
alter table public.ivon_users
  add column container_settings jsonb not null default '{}'::jsonb,
  add column container_connections jsonb not null default '{}'::jsonb,
  add constraint ivon_container_settings_object check (jsonb_typeof(container_settings) = 'object'),
  add constraint ivon_container_connections_object check (jsonb_typeof(container_connections) = 'object');

create table public.ivon_container_grants (
  session_id uuid primary key,
  user_id uuid not null references public.ivon_users(id) on delete cascade,
  token_hash text not null unique check (token_hash ~ '^[0-9a-f]{64}$'),
  expires_at timestamptz not null,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);
create index ivon_container_grants_user_id_idx on public.ivon_container_grants(user_id);
alter table public.ivon_container_grants enable row level security;
alter table public.ivon_users enable row level security;
revoke all on public.ivon_container_grants from public, anon, authenticated;
revoke all on public.ivon_users from public, anon, authenticated;
grant select, insert, update, delete on public.ivon_container_grants to service_role;
grant select, insert, update, delete on public.ivon_users to service_role;

-- Update one connection atomically, preserving concurrent edits to other accounts.
create function public.ivon_set_container_connection(owner_id uuid, connection_id text, connection jsonb)
returns void language sql security invoker set search_path = '' as $$
  update public.ivon_users
  set container_connections = case when connection is null
    then container_connections - connection_id
    else jsonb_set(container_connections, array[connection_id], connection, true) end,
    updated_at = now()
  where id = owner_id;
$$;
revoke all on function public.ivon_set_container_connection(uuid, text, jsonb) from public, anon, authenticated;
grant execute on function public.ivon_set_container_connection(uuid, text, jsonb) to service_role;

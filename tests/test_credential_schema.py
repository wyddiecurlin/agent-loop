# AI_OWNED
"""Exercise the migration against real PostgreSQL in the test container."""
import os
import pwd
import subprocess
import tempfile
from pathlib import Path


def main():
    binaries = next(Path('/usr/lib/postgresql').glob('*/bin'))
    user = pwd.getpwnam('postgres')
    with tempfile.TemporaryDirectory(prefix='credential-schema-') as temporary:
        os.chmod(temporary, 0o777)
        data = temporary + '/data'
        def pg(command, *args):
            return subprocess.run([str(binaries / command), *args], check=True,
                                  user=user.pw_uid, group=user.pw_gid, capture_output=True, text=True)
        pg('initdb', '-D', data, '--auth=trust', '--no-locale')
        pg('pg_ctl', '-D', data, '-l', temporary + '/postgres.log', '-o', '-k ' + temporary + " -c listen_addresses=''", '-w', 'start')
        try:
            schema = Path('/app/supabase/migrations/202609100001_container_credentials.sql').read_text()
            sql = '''
create role anon;
create role authenticated;
create role service_role bypassrls;
create schema auth;
create table auth.users(id uuid primary key);
create table public.ivon_users(id uuid primary key references auth.users(id), updated_at timestamptz default now());
''' + schema + '''
insert into auth.users values ('11111111-1111-4111-8111-111111111111');
insert into public.ivon_users(id) select id from auth.users;
set role anon;
do $$begin
  begin perform * from public.ivon_users; raise exception 'anon read succeeded'; exception when insufficient_privilege then null; end;
  begin perform * from public.ivon_container_grants; raise exception 'anon grant read succeeded'; exception when insufficient_privilege then null; end;
end$$;
reset role;
set role authenticated;
do $$begin
  begin perform * from public.ivon_users; raise exception 'client secret read succeeded'; exception when insufficient_privilege then null; end;
  begin perform public.ivon_set_container_connection('11111111-1111-4111-8111-111111111111','openai','{}'); raise exception 'client write succeeded'; exception when insufficient_privilege then null; end;
end$$;
reset role;
set role service_role;
insert into public.ivon_container_grants(session_id,user_id,token_hash,expires_at)
values ('22222222-2222-4222-8222-222222222222','11111111-1111-4111-8111-111111111111',repeat('a',64),now()+interval '15 minutes');
select public.ivon_set_container_connection('11111111-1111-4111-8111-111111111111','openai','{"secret":{"ciphertext":"encrypted"}}');
select public.ivon_set_container_connection('11111111-1111-4111-8111-111111111111','brave','{"secret":{"ciphertext":"another"}}');
do $$begin
  if not exists(select 1 from public.ivon_users where container_connections ? 'openai' and container_connections ? 'brave') then raise exception 'connection update lost data'; end if;
  begin update public.ivon_container_grants set token_hash='raw-token'; raise exception 'raw token accepted'; exception when check_violation then null; end;
end$$;
reset role;
'''
            result = subprocess.run([str(binaries / 'psql'), '-h', temporary, '-U', 'postgres', '-v', 'ON_ERROR_STOP=1'], input=sql, text=True, capture_output=True)
            if result.returncode:
                raise AssertionError(result.stderr)
            print('PASS: PostgreSQL migration, role isolation, hash constraint, and atomic connection updates')
        finally:
            pg('pg_ctl', '-D', data, '-m', 'immediate', '-w', 'stop')
            subprocess.run(['rm', '-rf', data], user=user.pw_uid, group=user.pw_gid, check=True)


if __name__ == '__main__':
    main()

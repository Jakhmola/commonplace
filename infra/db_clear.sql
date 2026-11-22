-- Empties every app table but keeps the schema and the migration history.
DO $$
DECLARE r RECORD;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables
           WHERE schemaname = 'public' AND tablename <> 'alembic_version' LOOP
    EXECUTE 'TRUNCATE ' || quote_ident(r.tablename) || ' CASCADE';
  END LOOP;
END $$;

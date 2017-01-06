function testpostgres()
    println("Testing basic connection")
    PostgresType = PostgreSQL.PostgresType
    conn = connect(Postgres, "localhost", "postgres")
    println("Connection made, making temporary table")
    run(conn, """CREATE TEMPORARY TABLE foobar (foo INTEGER PRIMARY KEY, bar DOUBLE PRECISION, 
        foobar CHARACTER VARYING);""")
    stmt = prepare(conn, "INSERT INTO foobar (foo, bar, foobar) VALUES (\$1, \$2, \$3);")
    println("Testing statement types as result of `prepare`")
    @test stmt.paramtypes[1] == PostgresType{:int4}
    @test stmt.paramtypes[2] == PostgresType{:float8}
    @test stmt.paramtypes[3] == PostgresType{:varchar}
end

testpostgres()

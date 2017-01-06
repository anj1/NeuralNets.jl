function test_connection()
    libpq = PostgreSQL.libpq

    println("Testing basic connections")
    conn = connect(Postgres, "localhost", "postgres")
    @test isa(conn, PostgreSQL.PostgresDatabaseHandle)
    @test conn.status == PostgreSQL.CONNECTION_OK
    @test errcode(conn) == PostgreSQL.CONNECTION_OK
    @test !conn.closed
    @test bytestring(libpq.PQdb(conn.ptr)) == "postgres"
    @test bytestring(libpq.PQuser(conn.ptr)) == "postgres"
    @test bytestring(libpq.PQport(conn.ptr)) == "5432"
    
    disconnect(conn)
    println("Basic connection function passed")
    @test conn.closed

    println("Trying connection with do block notation")
    conn = connect(Postgres, "localhost", "postgres") do conn
        @test isa(conn, PostgreSQL.PostgresDatabaseHandle)
        @test conn.status == PostgreSQL.CONNECTION_OK
        @test errcode(conn) == PostgreSQL.CONNECTION_OK
        @test !conn.closed
        return conn
    end
    println("Do block notation passed")
    @test conn.closed

    println("Testing connection with DSN string")
    conn = connect(Postgres, "postgres://postgres@localhost:5432", "postgres") do conn
        @test isa(conn, PostgreSQL.PostgresDatabaseHandle)
        @test conn.status == PostgreSQL.CONNECTION_OK
        @test errcode(conn) == PostgreSQL.CONNECTION_OK
        @test !conn.closed
        return conn
    end
    @test conn.closed
    println("DSN connection passed in a do block")
end

test_connection()

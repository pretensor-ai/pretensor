def build_clause(column):
    return "WHERE id = 1 AND name = " + column


result = build_clause("users")

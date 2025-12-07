def validate_sql(state):
    q = state.get("sql_query", "")
    res = state.get("sql_result", {})
    rows= res.get("rows", [])

    # Empty SQL
    if not q or not q.strip():
        return "empty_sql"

    # empty results
    if len(rows) == 0:
        return "empty_rows"
    
    # for futurstic checks (execution errors, etc.)

    return None

def validate_output(state):
    answer = state.get("final_answer", None)
    citations = state.get("citations", [])

    # No answer
    if answer is None or answer == "":
        return "empty_answer"

    # Citations invalid
    if isinstance(citations, list) and len(citations) == 0:
        return "missing_citations"

    return None

def repair_issue(state):
    mode = state.get("mode", "hybrid")

    # SQL-related validations (sql & hybrid modes)
    if mode in ("sql", "hybrid"):
        sql_issue = validate_sql(state)
        
        # if sql_issue == "empty_sql" or sql_issue == "empty_rows":
        if sql_issue:
            return "nl2sql"
    
    # Output validation (for ALL modes)
    output_issue = validate_output(state)

    if output_issue:
        return "synth"

    print("\n No issues detected in output nor sql. \n")
    # Everything is valid
    return None

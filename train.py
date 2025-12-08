data = [
    # RAG examples (document/marketing/etc)
    ("Show marketing calendar for June", "rag"),
    ("What does docs/marketing_calendar.md say about beverages?", "rag"),
    ("Find policies in docs/product_policy.md", "rag"),
    ("Search product_policy return window for dairy", "rag"),
    ("Retrieve marketing_calendar::chunk0 about Winter Classics", "rag"),
    ("Look up KPI definitions in docs/kpi_definitions.md", "rag"),
    ("Give me documentation on categories in docs/catalog.md", "rag"),
    ("Find notes in marketing_calendar for 1997 Winter", "rag"),
    ("Which documents mention Beverages?", "rag"),
    ("Show doc chunks referencing AOV", "rag"),

    # SQL examples (explicit DB/KPI/orders queries)
    ("Total revenue between 1997-06-01 and 1997-06-30", "sql"),
    ("How many orders in June 1997?", "sql"),
    ("AOV for Beverages category last month", "sql"),
    ("Write SQL to compute gross margin by category", "sql"),
    ("Orders grouped by customer region for 1997-12", "sql"),
    ("Return OrderID, Quantity where discount > 0.1", "sql"),
    ("Show top 5 products by revenue", "sql"),
    ("Run SQL: SELECT SUM(UnitPrice*Quantity) FROM OrderDetails", "sql"),
    ("I want sales by category and date range", "sql"),

    # Hybrid / ambiguous examples
    ("Explain sales trends and include docs/marketing_calendar.md if needed", "hybrid"),
    ("Combine doc guidance and DB numbers for holiday promotions", "hybrid"),
    ("Use docs and DB to compute projected orders for Winter Classics", "hybrid"),
    ("Idea: use marketing docs and actual orders to recommend promotions", "hybrid"),
    ("Should we trust docs or DB for AOV discrepancies?", "hybrid"),
    ("Combine KPIs from docs and SQL to produce a recommendation", "hybrid"),
    ("Show documents that explain AOV and then run SQL for numbers", "hybrid"),
    ("Use docs for policy and DB for counts (e.g., returns)", "hybrid"),
    ("Prepare report using docs and underlying DB", "hybrid"),
    ("Fetch doc chunks then compute the KPI from SQL", "hybrid"),
]




def generate_pretty_html_table_custom(df, title=None, css_styles=None):
    """
    Generates a styled HTML table from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to render as HTML.
        title (str, optional): An optional title to include above the table.
        css_styles (str, optional): Custom CSS styles as a string.

    Returns:
        str: A string containing the HTML for the styled table.
    """
    # Define default styles if none provided
    if css_styles is None:
        css_styles = """
        <style>
            table {
                border-spacing: 0;
                border-collapse: collapse;
                width: max-content;
                max-width: 100%;
                overflow: auto;
                border-color: gray;
            }
            
            /* Row styling (alternating colors) */
            tr:nth-child(2n) {
                background-color: #f6f8fa;
            }
            
            tr {
                background-color: #ffffff;
                border-top: 1px solid #d0d7deb3;
            }
            
            /* Cell styling */
            td {
                padding: 6px 13px;
                border: 1px solid #d0d7de;
                text-align: left;
                vertical-align: inherit;
            }
            
            /* Header cell styling */
            th {
                padding: 6px 13px;
                border: 1px solid #d0d7de;
                background-color: #f0f0f0;
                text-align: left;
            }
            
            p {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
              font-size: 16px;
              line-height: 1.6;
              color: #24292e; /* Dark grey text */
              margin: 1em 0;   /* Add spacing above and below */
              word-wrap: break-word; /* Prevent text overflow */
            }
            
            p code {
              background-color: rgba(27,31,35,0.05); /* Light grey background for inline code */
              padding: 0.2em 0.4em;
              font-size: 85%; /* Slightly smaller font size */
              border-radius: 6px;
              font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
              color: #e83e8c; /* Code color */
            }
        </style>
        """

    # Convert the DataFrame to HTML
    table_html = df.to_html(index=False, escape=False)

    # Add optional title
    if title:
        table_html = f"<caption>{title}</caption>\n" + table_html

    # Combine styles and table HTML
    full_html = f"{css_styles}\n<div>{table_html}</div>"
    return full_html


def generate_pretty_html_table_default(df):
    styled_df = (
        df.style
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("border-spacing", "0"),
                        ("border-collapse", "collapse"),
                        ("width", "max-content"),
                        ("max-width", "100%"),
                        ("overflow", "auto"),
                        ("border-color", "gray"),
                    ]
                },
                # Row styling (alternating colors)
                {
                    "selector": "tr:nth-child(2n)",
                    "props": [("background-color", "#f6f8fa")]
                },
                {
                    "selector": "tr",
                    "props": [
                        ("background-color", "#ffffff"),
                        ("border-top", "1px solid #d0d7deb3")
                    ]
                },
                # Cell styling
                {
                    "selector": "td",
                    "props": [
                        ("padding", "6px 13px"),
                        ("border", "1px solid #d0d7de"),
                        ("text-align", "left"),
                        ("vertical-align", "inherit"),
                    ]
                },
                # Optional: Add styles for header cells if needed
                {
                    "selector": "th",
                    "props": [
                        ("padding", "6px 13px"),
                        ("border", "1px solid #d0d7de"),
                        ("background-color", "#f0f0f0"),
                        ("text-align", "left"),
                    ]
                }
            ]
        )
        .set_properties(**{"text-align": "left"})
    )

    return styled_df.to_html()
#!/usr/bin/env python3
"""
Generate static Swagger documentation from FastAPI app
"""

import json
import os
import sys
from pathlib import Path

# Import your FastAPI app
sys.path.insert(0, os.path.abspath("src"))
from start_server import app

# Create docs directory
output_dir = Path("dist")
output_dir.mkdir(exist_ok=True)

# Generate OpenAPI JSON
print("Generating OpenAPI schema from FastAPI app...")

openapi_schema = app.openapi()

with open(output_dir / "openapi.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)

print(f"OpenAPI schema saved to {output_dir}/openapi.json")

# Create Swagger UI HTML
swagger_html = """<!DOCTYPE html>
<html>
  <head>
    <title>Language Learning API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
      *, *:before, *:after { box-sizing: inherit; }
      body { margin: 0; padding: 0; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js" charset="UTF-8"></script>
    <script>
      window.onload = function() {
        const ui = SwaggerUIBundle({
          url: "openapi.json",
          dom_id: '#swagger-ui',
          deepLinking: true,
          presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIBundle.SwaggerUIStandalonePreset
          ],
          layout: "BaseLayout",
          supportedSubmitMethods: []  // Disable "Try it out" button since this is static
        });
        window.ui = ui;
      };
    </script>
  </body>
</html>"""

with open(output_dir / "swagger.html", "w") as f:
    f.write(swagger_html)

# Create ReDoc HTML (alternative documentation UI)
redoc_html = """<!DOCTYPE html>
<html>
  <head>
    <title>Language Learning API Documentation (ReDoc)</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
      body { margin: 0; padding: 0; }
    </style>
  </head>
  <body>
    <redoc spec-url="openapi.json"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
  </body>
</html>"""

with open(output_dir / "index.html", "w") as f:
    f.write(redoc_html)

print("Documentation generation complete!")

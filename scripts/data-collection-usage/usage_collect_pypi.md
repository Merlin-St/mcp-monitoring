PyPI Data Collection:
To get PyPI download statistics, run this query in Google Cloud Console BigQuery web UI:

SELECT 
    file.project AS package_name,
    FORMAT_DATE('%Y-%m', DATE_TRUNC(DATE(timestamp), MONTH)) AS month,
    COUNT(*) AS downloads
FROM 
    `bigquery-public-data.pypi.file_downloads`
WHERE 
    LOWER(file.project) LIKE '%mcp%'
    AND DATE(timestamp) >= '2024-11-01'
    AND DATE(timestamp) < '2025-09-01'
GROUP BY 
    package_name, 
    month
ORDER BY 
    package_name, 
    month

Export the results as JSON and save as 'usage_bigquery_webresults_pypi.json'
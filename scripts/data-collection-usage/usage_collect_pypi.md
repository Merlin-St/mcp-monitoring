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


Alternative with metadata: 

WITH monthly_downloads AS (
  SELECT
    file.project AS name,
    FORMAT_DATE('%Y-%m', DATE(file.timestamp)) AS month,
    COUNT(*) AS downloads
  FROM
    `bigquery-public-data.pypi.file_downloads` AS file
  WHERE
    DATE(file.timestamp) BETWEEN '2024-11-01' AND '2025-09-30'
  GROUP BY
    name,
    month
),

latest_metadata AS (
  SELECT
    m.name,
    m.summary,
    m.description,
    m.description_content_type,
    m.keywords,
    m.author,
    m.author_email,
    m.maintainer_email,
    m.classifiers,
    m.project_urls,
    m.upload_time,
    ROW_NUMBER() OVER (PARTITION BY m.name ORDER BY m.version DESC) AS rn
  FROM
    `bigquery-public-data.pypi.distribution_metadata` AS m
)

SELECT
  md.name,
  md.month,
  md.downloads AS monthly_downloads,
  lm.summary,
  lm.description,
  lm.description_content_type AS `description-content-type`,
  lm.keywords,
  lm.classifiers,
  lm.author,
  lm.author_email,
  lm.maintainer_email,
  lm.project_urls AS `Project-URLs`,
  lm.upload_time
FROM
  monthly_downloads AS md
LEFT JOIN
  latest_metadata AS lm
ON
  md.name = lm.name
  AND lm.rn = 1  -- only latest version
WHERE
  CONTAINS_SUBSTR(IFNULL(lm.name, ''), 'mcp')
  OR CONTAINS_SUBSTR(IFNULL(lm.summary, ''), 'mcp')
  OR CONTAINS_SUBSTR(IFNULL(lm.description, ''), 'mcp')
  OR CONTAINS_SUBSTR(IFNULL(lm.keywords, ''), 'mcp')
ORDER BY
  md.name, md.month DESC;


Latest option 3: With country code

WITH monthly_downloads AS (
  SELECT
    file.project AS name,
    file.country_code,
    FORMAT_DATE('%Y-%m', DATE(file.timestamp)) AS month,
    COUNT(*) AS downloads
  FROM
    `bigquery-public-data.pypi.file_downloads` AS file
  WHERE
    DATE(file.timestamp) BETWEEN '2024-11-01' AND '2025-10-31'
  GROUP BY
    name,
    country_code,  -- Added this line
    month
),

latest_metadata AS (
  SELECT
    m.name,
    m.summary,
    m.description,
    m.description_content_type,
    m.keywords,
    m.author,
    m.author_email,
    m.maintainer_email,
    m.classifiers,
    m.project_urls,
    m.upload_time,
    ROW_NUMBER() OVER (PARTITION BY m.name ORDER BY m.version DESC) AS rn
  FROM
    `bigquery-public-data.pypi.distribution_metadata` AS m
)

SELECT
  md.name,
  md.country_code,
  md.month,
  md.downloads AS monthly_downloads,
  lm.summary,
  lm.description,
  lm.description_content_type AS `description-content-type`,
  lm.keywords,
  lm.classifiers,
  lm.author,
  lm.author_email,
  lm.maintainer_email,
  lm.project_urls AS `Project-URLs`,
  lm.upload_time
FROM
  monthly_downloads AS md
LEFT JOIN
  latest_metadata AS lm
ON
  md.name = lm.name
  AND lm.rn = 1  -- only latest version
WHERE
  CONTAINS_SUBSTR(LOWER(IFNULL(lm.name, '')), 'mcp')
  OR CONTAINS_SUBSTR(LOWER(IFNULL(lm.summary, '')), 'mcp')
  OR CONTAINS_SUBSTR(LOWER(IFNULL(lm.description, '')), 'mcp')
  OR CONTAINS_SUBSTR(LOWER(IFNULL(lm.keywords, '')), 'mcp')
ORDER BY
  md.name, md.month DESC, md.country_code;
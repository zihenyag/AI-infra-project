WITH ranked AS (
 SELECT baseline, point, ttft_ms,
 ROW_NUMBER() OVER(PARTITION BY baseline, point ORDER BY ttft_ms) rn,
 COUNT(*) OVER(PARTITION BY baseline, point) n FROM requests
), ttft AS (
 SELECT baseline, point, AVG(ttft_ms) ttft_ms_p50 FROM ranked
 WHERE rn IN ((n+1)/2,(n+2)/2) GROUP BY baseline, point
), itl_ranked AS (
 SELECT baseline, point, value_ms,
 ROW_NUMBER() OVER(PARTITION BY baseline, point ORDER BY value_ms) rn,
 COUNT(*) OVER(PARTITION BY baseline, point) n FROM itls
), itl AS (
 SELECT baseline, point, AVG(value_ms) itl_ms_p50 FROM itl_ranked
 WHERE rn IN ((n+1)/2,(n+2)/2) GROUP BY baseline, point
), totals AS (
 SELECT baseline, point, COUNT(*) completed, SUM(prompt_tokens) prompt_tokens,
 SUM(output_tokens) output_tokens, SUM(cached_tokens) cached_tokens,
 AVG(prompt_tokens) actual_isl_mean, AVG(client_wait_s) client_wait_s_mean
 FROM requests GROUP BY baseline, point
)
SELECT totals.*, ttft.ttft_ms_p50, itl.itl_ms_p50, points.qps, points.duration_s,
 points.label, 1.0*totals.cached_tokens/totals.prompt_tokens cache_hit_ratio,
 1.0*totals.output_tokens/points.duration_s output_token_throughput,
 CASE totals.baseline WHEN 'S0' THEN 'solid' WHEN 'S1' THEN 'dashed' ELSE 'dotted' END line_style
FROM totals JOIN ttft USING(baseline,point) JOIN itl USING(baseline,point)
JOIN points USING(baseline,point)
ORDER BY points.ordering,totals.baseline;

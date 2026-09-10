SELECT baseline, ttft_ms,
 1.0*ROW_NUMBER() OVER(PARTITION BY baseline ORDER BY ttft_ms)/COUNT(*) OVER(PARTITION BY baseline) cdf,
 COUNT(*) OVER(PARTITION BY baseline) n,
 CASE baseline WHEN 'S0' THEN 'solid' WHEN 'S1' THEN 'dashed' ELSE 'dotted' END line_style
FROM requests WHERE point='pressure_4k_q1' ORDER BY baseline,ttft_ms;

# Primary publication results

These aggregate-only tables were rendered from verified inference artifact
`0fe2e85baefd1befc22476c92b7595659aeac4528e9899a191b0abb09c57b2f3`. The artifact itself is not
committed because it binds licensed targets and predictions. Lower MAPE and RMSE are better; a
negative paired contrast favors the model on the left. `Reject` is the predeclared two-sided Holm
decision over six model pairs and two primary metrics.

| Variant | N | Date panels | MAPE (%) [95% CI] | RMSE [95% CI] |
|---|---:|---:|---:|---:|
| Cash jump (liquidator) | 678122 | 2400 | 4.499252 [3.559918, 5.125044] | 0.517789 [0.306943, 0.716231] |
| Escrowed cash (spot matched) | 678122 | 2400 | 4.488501 [3.530267, 5.122251] | 0.525518 [0.306907, 0.734032] |
| Forecast continuous yield | 678122 | 2400 | 5.179783 [4.305128, 5.769923] | 0.550468 [0.351751, 0.727428] |
| Historical continuous yield | 678122 | 2400 | 5.725587 [4.874838, 6.311240] | 0.560446 [0.367044, 0.730793] |

| Left - right | Metric | Estimate [95% CI] | p | Holm p | Reject |
|---|---|---:|---:|---:|:---:|
| Cash jump (liquidator) - Escrowed cash (spot matched) | MAPE (%) | 0.010752 [-0.008720, 0.033901] | 0.325200 | 0.325200 | no |
| Cash jump (liquidator) - Escrowed cash (spot matched) | RMSE | -0.007729 [-0.018179, 0.000525] | 0.114100 | 0.237000 | no |
| Cash jump (liquidator) - Forecast continuous yield | MAPE (%) | -0.680531 [-0.783162, -0.579421] | 0.000050 | 0.000600 | yes |
| Cash jump (liquidator) - Forecast continuous yield | RMSE | -0.032679 [-0.045978, -0.009952] | 0.000200 | 0.001400 | yes |
| Cash jump (liquidator) - Historical continuous yield | MAPE (%) | -1.226335 [-1.395186, -1.056169] | 0.000050 | 0.000600 | yes |
| Cash jump (liquidator) - Historical continuous yield | RMSE | -0.042658 [-0.062415, -0.012517] | 0.000600 | 0.003600 | yes |
| Escrowed cash (spot matched) - Forecast continuous yield | MAPE (%) | -0.691283 [-0.813276, -0.576299] | 0.000050 | 0.000600 | yes |
| Escrowed cash (spot matched) - Forecast continuous yield | RMSE | -0.024951 [-0.045630, 0.007753] | 0.079000 | 0.237000 | no |
| Escrowed cash (spot matched) - Historical continuous yield | MAPE (%) | -1.237086 [-1.419606, -1.055723] | 0.000050 | 0.000600 | yes |
| Escrowed cash (spot matched) - Historical continuous yield | RMSE | -0.034929 [-0.061582, 0.004847] | 0.051050 | 0.204200 | no |
| Forecast continuous yield - Historical continuous yield | MAPE (%) | -0.545804 [-0.640170, -0.449070] | 0.000050 | 0.000600 | yes |
| Forecast continuous yield - Historical continuous yield | RMSE | -0.009978 [-0.017632, -0.001144] | 0.015350 | 0.076750 | no |

The corresponding booktabs fragments are [primary models](primary-models.tex) and
[primary contrasts](primary-contrasts.tex). Re-render either format with:

```shell
discrete-dividend-study render-publication-inference-table \
  --artifact <verified-inference.safetensors> \
  --table primary-models \
  --format markdown
```

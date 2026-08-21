# Elo Merchant Rank6 Release

本仓库用于复现 `Private RMSE 3.59428`。

```bash
python -m pip install -r requirements.txt
python src/export_submission.py
python src/verify_submission.py
```

运行后会生成 `submission/submission_rank6_3.59428.csv.gz`，直接提交即可。

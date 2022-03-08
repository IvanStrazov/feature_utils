# Feature Utils

```
import pandas as pd
from feature_utils import FeatureMaker

df = pd.DataFrame({"A": [0,1,1,2], "B": ["a","a","b","c"]})

fm = FeatureMaker()
fm.target_cat_encode(df, ["B"], ["A"], ["mean", "q_25", "classic", "classic_ts", "diff_abs_1", "iqr_25_75"])

```

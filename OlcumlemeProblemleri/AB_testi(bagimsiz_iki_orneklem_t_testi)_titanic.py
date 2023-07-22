#Uygulama: titanic kadın ve erkek yolcuların yaş ortalamaları arasında anlamlı bir fark var mıdır?

import seaborn as sns
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

df_ = sns.load_dataset("titanic")
df = df_.copy()

df.head()
df.isnull().any()
df.isnull().sum()

#İlk olarak kabaca kadın ve erkeklerin yaşları arasında fark var mı diye kontrol ederiz:
df.groupby("sex").agg({"age":"mean"})

"""
female 27.91571
male   30.72664

Görüldüğü üzere matematiksel olarak yaşların ortalaması arasında anlamlı bir fark olduğu gözüküyor.
Fakat bu fark şans eseri mi meydana geldi bilemiyoruz.
Bu durumu net anlamak adına istatistiksel anlamda da bir fark olup olmadığını kontorl etmemiz gerekir.
"""

# 1. Hipotez kurulur:
# H0: M1 = M2
# titanic kadın ve erkek yaş ortalamaları arasında istatistiksel olarak anlamlı fark yoktur.

# H1: M1 != M2 (ana kütle ortalamaların temsilleridir.)
# titanic kadın ve erkek yaş ortalamaları arasında istatistiksel olarak anlamlı fark vardır.

# 2. Varsayımların Kontrolü Yapılır:
# 2.1. Normallik Varsayımı Kontrolü:

# H0: Normallik dağılımları sağlanmaktadır.
# H1: Normallik dağılımları sağlanmamaktadır.

#Dikkat: age değişkeninde eksik değerler mevcut, onları silmemiz gerekecek!
#1. grup için normallik varsayımını testi:
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

# pvalue= 0.0071 < 0.05 (1. grup için Normallik var sayımı sağlanmaz.)

#2. grup için normallik varsayımını testi:
test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#pvalue = 0.0000 < 0.05 (2. grup için Normallik var sayımı sağlanmaz.)


# 3.2. Varsayımlar sağlanmadığı için mannwhitneyu testi(non-parametrik test) yapılır:
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#pvalue = 0.0261 < 0.05 H0 RED
#Bu şu anlama gelir. İstatistiki olarak da anlamlı bir farkın olduğunu bildirir.


# 90 2000 matematiksel olarak bir fark olduğunu görüyoruz.
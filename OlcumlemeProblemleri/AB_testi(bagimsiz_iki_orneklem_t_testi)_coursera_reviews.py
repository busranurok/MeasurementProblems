import seaborn as sns
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float.format", lambda x: "%.5f" % x)

df_ = pd.read_csv("/Users/hbo/PycharmProjects/OlcumlemeProblemleri/dataset/course_reviews.csv")
df = df_.copy()

df.head()
df.isnull().any()
df.isnull().sum()

#Uygulama: Kursu izleyenler ile izlemeyenlerin ortalama puanlarında istatistiksel olarak anlamlı bir fark var mı?

df[df["Progress"] > 75]["Rating"].mean()

df[df["Progress"] < 10]["Rating"].mean()


# 1. grup için normallik varsayımı testi yapılır:
test_stat, pvalue = shapiro(df[df["Progress"] > 75]["Rating"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#p-value = 0.0000 < 0.05 HO RED

# 2. grup için normallik varsayımı testi yapılır:
test_stat, pvalue = shapiro(df[df["Progress"] < 10]["Rating"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))
#p-value = 0.0000 < 0.05 HO RED
#Normallik varsayımı sağlanmıyor.

# 3. non-parametrik test(mannwhitneyu testi)
test_stat, pvalue = mannwhitneyu(df[df["Progress"] > 75]["Rating"],
                                 df[df["Progress"] < 10]["Rating"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#p-value = 0.0000 < 0.05 HO RED

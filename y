warning: in the working copy of 'notebooks/01_data_transforamtion.ipynb', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/.gitignore b/.gitignore[m
[1mindex 53a15ea..0c7daef 100644[m
[1m--- a/.gitignore[m
[1m+++ b/.gitignore[m
[36m@@ -2,9 +2,7 @@[m [mlogs/[m
 artifact/[m
 src.egg-info/[m
 template.py[m
[31m-notebooks/02_EDA.ipynb[m
[31m-notebooks/FeatureENG.txt[m
[31m-notebooks/cleanining_application_train.txt[m
[32m+[m
 template.py[m
 src\__pycache__[m
 **\__pycache__\[m
[1mdiff --git a/dvc.lock b/dvc.lock[m
[1mindex 5c2c520..4650aad 100644[m
[1m--- a/dvc.lock[m
[1m+++ b/dvc.lock[m
[36m@@ -68,8 +68,8 @@[m [mstages:[m
       nfiles: 7[m
     - path: src/components/03_data_transformation.py[m
       hash: md5[m
[31m-      md5: cffb0115cc76aa86a47bc56f1f27c42a[m
[31m-      size: 72466[m
[32m+[m[32m      md5: 69610e5b84372cca0c09d7cc5abfa39e[m
[32m+[m[32m      size: 73406[m
     - path: src/constants/data_transformation_constant.py[m
       hash: md5[m
       md5: 94357cafb211de1c301265c9f80a81b9[m
[36m@@ -85,6 +85,6 @@[m [mstages:[m
     outs:[m
     - path: artifact/interim[m
       hash: md5[m
[31m-      md5: 41d0c0b584d116c28fea1f7896914aed.dir[m
[31m-      size: 231798359[m
[32m+[m[32m      md5: 6c036c8a6597079c11c922269de0aca8.dir[m
[32m+[m[32m      size: 234323804[m
       nfiles: 1[m
[1mdiff --git a/notebooks/01_data_transforamtion.ipynb b/notebooks/01_data_transforamtion.ipynb[m
[1mindex b0aea52..6c1fe63 100644[m
[1m--- a/notebooks/01_data_transforamtion.ipynb[m
[1m+++ b/notebooks/01_data_transforamtion.ipynb[m
[36m@@ -2,7 +2,7 @@[m
  "cells": [[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": null,[m
[32m+[m[32m   "execution_count": 1,[m
    "metadata": {},[m
    "outputs": [],[m
    "source": [[m
[36m@@ -23,7 +23,7 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": null,[m
[32m+[m[32m   "execution_count": 13,[m
    "metadata": {},[m
    "outputs": [],[m
    "source": [[m
[36m@@ -32,7 +32,7 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": null,[m
[32m+[m[32m   "execution_count": 14,[m
    "metadata": {},[m
    "outputs": [[m
     {[m
[36m@@ -1064,7 +1064,7 @@[m
        "4                        0.0                         0.0  "[m
       ][m
      },[m
[31m-     "execution_count": 3,[m
[32m+[m[32m     "execution_count": 14,[m
      "metadata": {},[m
      "output_type": "execute_result"[m
     }[m
[36m@@ -1195,6 +1195,11 @@[m
     "application_train[application_train['YEARS_EMPLOYED'] <-999][['YEARS_EMPLOYED','DAYS_EMPLOYED']].head()"[m
    ][m
   },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "metadata": {},[m
[32m+[m[32m   "source": [][m
[32m+[m[32m  },[m
   {[m
    "cell_type": "code",[m
    "execution_count": null,[m
[36m@@ -2410,7 +2415,7 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 5,[m
[32m+[m[32m   "execution_count": 116,[m
    "metadata": {},[m
    "outputs": [[m
     {[m
[36m@@ -2595,7 +2600,7 @@[m
        "248486  Consumer credit                -155          0.0  "[m
       ][m
      },[m
[31m-     "execution_count": 5,[m
[32m+[m[32m     "execution_count": 116,[m
      "metadata": {},[m
      "output_type": "execute_result"[m
     }[m
[36m@@ -2606,84 +2611,96 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 27,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [],[m
[31m-   "source": [[m
[31m-    "##### drop CREDIT_CURRENCY\n",[m
[31m-    "# can create is he is taken the loan from another country \n",[m
[31m-    "# has_taken_loan_foreign\n",[m
[31m-    "# this feature have 99% currency 1 values so this is low variance feature. drop it if it doesnt add any value."[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "markdown",[m
[31m-   "metadata": {},[m
[31m-   "source": [[m
[31m-    "#### CREDIT_ACTIVE"[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 50,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [],[m
[31m-   "source": [[m
[31m-    "# no of active credit..<br>\n",[m
[31m-    "# no of closed credit..<br>\n",[m
[31m-    "# HAS_BAD_LOAN.."[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 51,[m
[32m+[m[32m   "execution_count": 8,[m
    "metadata": {},[m
    "outputs": [[m
     {[m
      "data": {[m
       "text/plain": [[m
[31m-       "CREDIT_ACTIVE\n",[m
[31m-       "Closed      1079273\n",[m
[31m-       "Active       630607\n",[m
[31m-       "Sold           6527\n",[m
[31m-       "Bad debt         21\n",[m
[32m+[m[32m       "CREDIT_TYPE\n",[m
[32m+[m[32m       "Consumer credit                                 1251615\n",[m
[32m+[m[32m       "Credit card                                      402195\n",[m
[32m+[m[32m       "Car loan                                          27690\n",[m
[32m+[m[32m       "Mortgage                                          18391\n",[m
[32m+[m[32m       "Microloan                                         12413\n",[m
[32m+[m[32m       "Loan for business development                      1975\n",[m
[32m+[m[32m       "Another type of loan                               1017\n",[m
[32m+[m[32m       "Unknown type of loan                                555\n",[m
[32m+[m[32m       "Loan for working capital replenishment              469\n",[m
[32m+[m[32m       "Cash loan (non-earmarked)                            56\n",[m
[32m+[m[32m       "Real estate loan                                     27\n",[m
[32m+[m[32m       "Loan for the purchase of equipment                   19\n",[m
[32m+[m[32m       "Loan for purchase of shares (margin lending)          4\n",[m
[32m+[m[32m       "Interbank credit                                      1\n",[m
[32m+[m[32m       "Mobile operator loan                                  1\n",[m
        "Name: count, dtype: int64"[m
       ][m
      },[m
[31m-     "execution_count": 51,[m
[32m+[m[32m     "execution_count": 8,[m
      "metadata": {},[m
      "output_type": "execute_result"[m
     }[m
    ],[m
    "source": [[m
[31m-    "bureau['CREDIT_ACTIVE'].value_counts()"[m
[32m+[m[32m    "bureau['CREDIT_TYPE'].value_counts()"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "metadata": {},[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "#### NUM_CREDIT_CURRENCIES"[m
    ][m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 52,[m
[32m+[m[32m   "execution_count": 125,[m
    "metadata": {},[m
    "outputs": [],[m
    "source": [[m
[31m-    "crosstab = pd.crosstab(bureau['SK_ID_CURR'], bureau['CREDIT_ACTIVE'],dropna=False)\n",[m
[31m-    "if any(pd.isna(crosstab.columns)):\n",[m
[31m-    "    idx = crosstab[crosstab[np.nan]==1].index\n",[m
[31m-    "    crosstab.loc[idx] = np.nan"[m
[32m+[m[32m    "feature_df = bureau.groupby(by='SK_ID_CURR')['CREDIT_CURRENCY'].nunique().to_frame('NUM_CREDIT_CURRENCIES')\n",[m
[32m+[m[32m    "# this feature have 99% currency 1 values "[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "metadata": {},[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "#### TIME FRAMED FEATURE NUM_ACTIVE_CREDIT"[m
    ][m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 53,[m
[32m+[m[32m   "execution_count": 58,[m
    "metadata": {},[m
    "outputs": [],[m
    "source": [[m
[31m-    "#has bad loan\n",[m
[31m-    "crosstab['HAS_BAD_LOAN'] = ((crosstab['Bad debt'] > 0) | (crosstab['Sold']> 0)).astype(int)"[m
[32m+[m[32m    "\n",[m
[32m+[m[32m    "time_frames = [90, 180, 270, 360, 720]\n",[m
[32m+[m[32m    "\n",[m
[32m+[m[32m    "# empty dataframe to apend the feature into\n",[m
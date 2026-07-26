# Statistical Analysis Results

## 1. Bootstrap Confidence Intervals
- Pure model mean grokking step: 1337.5 (95% CI: [1275.0, 1406.2])

## 2. Hypothesis Testing
  - Not enough data to perform t-test across multiple seeds.

## 3. Correlation Analysis
- Spearman correlation between collapse level and grokking step:
  - correlation: 0.9435
  - p-value: 6.5052e-32

## 4. Multiple Regression: Predicting Grokking Step
### OLS Regression Results
```latex
\begin{center}
\begin{tabular}{lclc}
\toprule
\textbf{Dep. Variable:}    & grok\_step\_capped & \textbf{  R-squared:         } &     0.547   \\
\textbf{Model:}            &        OLS         & \textbf{  Adj. R-squared:    } &     0.532   \\
\textbf{Method:}           &   Least Squares    & \textbf{  F-statistic:       } &     37.44   \\
\textbf{Date:}             &  Sun, 26 Jul 2026  & \textbf{  Prob (F-statistic):} &  2.17e-11   \\
\textbf{Time:}             &      05:42:24      & \textbf{  Log-Likelihood:    } &   -713.08   \\
\textbf{No. Observations:} &           65       & \textbf{  AIC:               } &     1432.   \\
\textbf{Df Residuals:}     &           62       & \textbf{  BIC:               } &     1439.   \\
\textbf{Df Model:}         &            2       & \textbf{                     } &             \\
\textbf{Covariance Type:}  &     nonrobust      & \textbf{                     } &             \\
\bottomrule
\end{tabular}
\begin{tabular}{lcccccc}
                  & \textbf{coef} & \textbf{std err} & \textbf{t} & \textbf{P$> |$t$|$} & \textbf{[0.025} & \textbf{0.975]}  \\
\midrule
\textbf{level}    &    8.689e+04  &     1.01e+04     &     8.636  &         0.000        &     6.68e+04    &     1.07e+05     \\
\textbf{severity} &    -901.4195  &     7400.602     &    -0.122  &         0.903        &    -1.57e+04    &     1.39e+04     \\
\textbf{d\_model} &      53.6997  &       38.392     &     1.399  &         0.167        &      -23.045    &      130.444     \\
\bottomrule
\end{tabular}
\begin{tabular}{lclc}
\textbf{Omnibus:}       &  1.721 & \textbf{  Durbin-Watson:     } &    0.705  \\
\textbf{Prob(Omnibus):} &  0.423 & \textbf{  Jarque-Bera (JB):  } &    1.028  \\
\textbf{Skew:}          & -0.107 & \textbf{  Prob(JB):          } &    0.598  \\
\textbf{Kurtosis:}      &  3.577 & \textbf{  Cond. No.          } &     723.  \\
\bottomrule
\end{tabular}
%\caption{OLS Regression Results}
\end{center}

Notes: \newline
 [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.```

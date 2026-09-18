import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

F = '/home/claude/project/figs/'
D = '/home/claude/project/data/'

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=10, leading=14, spaceAfter=8, textColor=colors.black))
styles.add(ParagraphStyle(name='H1c', parent=styles['Heading1'], fontSize=15, spaceBefore=14, spaceAfter=8, textColor=colors.black))
styles.add(ParagraphStyle(name='H2c', parent=styles['Heading2'], fontSize=12.5, spaceBefore=10, spaceAfter=6, textColor=colors.black))
styles.add(ParagraphStyle(name='Caption', parent=styles['Normal'], fontSize=8.5, alignment=TA_CENTER, textColor=colors.black, spaceAfter=12, spaceBefore=2))
styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], fontSize=22, textColor=colors.black))
styles.add(ParagraphStyle(name='CoverSub', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER, textColor=colors.black, spaceBefore=10))
styles.add(ParagraphStyle(name='CoverName', parent=styles['Normal'], fontSize=13, alignment=TA_CENTER, textColor=colors.black, spaceBefore=6))
styles.add(ParagraphStyle(name='Note', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.black, backColor=colors.HexColor('#FFF5F5'), borderColor=colors.HexColor('#FC8181'), borderWidth=1, borderPadding=8, spaceAfter=10))

story = []

def H1(t): story.append(Paragraph(t, styles['H1c']))
def H2(t): story.append(Paragraph(t, styles['H2c']))
def P(t): story.append(Paragraph(t, styles['Justify']))
def IMG(path, width=15.5*cm, caption=None):
    story.append(Image(F+path, width=width, height=width*0.62))
    if caption: story.append(Paragraph(caption, styles['Caption']))
    else: story.append(Spacer(1,8))

def make_table(df, col_widths=None, fontsize=7.5):
    data = [list(df.columns)] + df.values.tolist()
    data = [[Paragraph(str(c), ParagraphStyle('c', fontSize=fontsize, leading=fontsize+2)) for c in row] for row in data]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#d9d9d9')),
        ('TEXTCOLOR',(0,0),(-1,0), colors.black),
        ('TEXTCOLOR',(0,1),(-1,-1), colors.black),
        ('FONTSIZE',(0,0),(-1,-1), fontsize),
        ('GRID',(0,0),(-1,-1), 0.4, colors.HexColor('#999999')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

# ---------------- COVER ----------------
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Sydney Housing Valuation - A Machine Learning Mini Project", styles['CoverTitle']))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph("Distinction Task", styles['CoverSub']))
story.append(Paragraph("Pranjal Gupta (S226230386)", styles['CoverName']))
story.append(Paragraph('GitHub Repository link: <link href="https://github.com/gpariee/Sydney-Housing-Valuation-" color="black">https://github.com/gpariee/Sydney-Housing-Valuation-</link>', styles['CoverName']))
story.append(Spacer(1, 6*cm))
story.append(PageBreak())

# ---------------- PART 1 ----------------
H1("Part 1 — Problem Definition and Data Collection")
H2("1.1 Problem and Suburb Selection")
P("""The task addressed in this project is a supervised regression problem: predicting the sale price of a residential
property in Sydney from its structural characteristics, location, and marketing description. Three suburbs were selected
to represent deliberately different segments of the Sydney housing market: <b>Mosman</b> (harbourside, high-income,
heritage and prestige housing, ~9 km from the CBD), <b>Marrickville</b> (inner-west, gentrifying, mixed housing stock of
Victorian/Federation cottages and modern apartments, ~7 km from the CBD), and <b>Blacktown</b> (outer-western Sydney,
family-oriented, larger and more affordable housing stock, ~30+ km from the CBD). These suburbs were chosen because they
differ substantially in median price, dwelling type mix, buyer demographic and distance-to-CBD — the combination lets a
single model be tested on very different price regimes rather than a narrow band of similar properties, which is a more
realistic and more difficult test of generalisation.""")

H2("1.2 Anticipated Data Quality Issues and Bias")
P("""Manual collection from listing portals carries several structural risks that this project's design tries to account
for. First, <b>survivorship bias</b>: only properties that successfully sold and had their result published are visible,
so off-market and passed-in sales are systematically excluded. Second, <b>missingness is not random</b> — land area is
routinely absent for strata (apartment/unit) listings, and older or lower-value listings are more likely to be missing
floor area, which is why this project explicitly flags imputed values with indicator columns rather than silently filling
them. Third, <b>agent description text is persuasive, not objective</b> — words like "exceptional" or "waterfront" are
marketing language, and any luxury/development-potential score derived from it (Section 2) captures marketing intensity,
which correlates with but does not equal true property quality. Finally, a 109-property sample across three suburbs is
small for a 15+ feature regression problem, which materially affects the bias–variance trade-off discussed in Part 3.""")
story.append(PageBreak())

# ---------------- PART 2 ----------------
H1("Part 2 — Data Understanding and Feature Engineering")
H2("2.1 Exploring the Distribution of Prices")
IMG('fig1_price_dist.png', caption="Figure 1. Sale price is strongly right-skewed (driven by Mosman's multi-million-dollar tail); "
    "the log transform is close to symmetric, motivating modelling on log(price).")
P("""Sale prices range from $490,000 (a Blacktown apartment) to $15,000,000 (a Mosman penthouse). The raw distribution
(Figure 1, left) is heavily right-skewed with a long tail of Mosman prestige sales; the log-transformed distribution
(right) is close to symmetric. All models in Part 3 are therefore trained on <b>log(sale price)</b> and converted back to
dollars for reporting, which is standard practice for price data spanning more than one order of magnitude.""")

IMG('fig2_price_by_suburb.png', caption="Figure 2. The three suburbs occupy almost non-overlapping price bands: "
    "Blacktown (median ≈ $0.9M), Marrickville (median ≈ $1.65M) and Mosman (median ≈ $5.0M).")
IMG('fig3_price_over_time.png', caption="Figure 3. Prices by settlement date. No strong seasonal trend is visible in this "
    "sample, though Mosman shows the widest month-to-month dispersion, consistent with a thin, heterogeneous prestige market.")

sub_summary = pd.read_csv(D+'raw_listings.csv')
sub_tab = sub_summary.groupby('Suburb')['SalePrice'].agg(Count='count', Mean='mean', Median='median', Std='std', Min='min', Max='max').reset_index()
for c in ['Mean','Median','Std','Min','Max']:
    sub_tab[c] = sub_tab[c].round(0).astype(int).map('${:,}'.format)
story.append(make_table(sub_tab, fontsize=8))
story.append(Spacer(1,10))

H2("2.2 Outliers")
P("""Using the 1.5×IQR rule within each suburb, only one clear outlier was identified — a $2.4M Blacktown house, well
above the suburb's $2.0M ceiling for comparable stock — while Marrickville and Mosman showed no formal outliers despite
their wide price ranges, because those ranges reflect genuinely heterogeneous stock (units through to waterfront estates)
rather than data errors. This single case was retained rather than removed, since it is a plausible (if unusually high)
transaction rather than a data-entry artefact, and Part 4 revisits large residuals directly through model errors rather
than through blanket removal.""")

H2("2.3 Feature Engineering and the Three Strongest Predictors")
P("""Before engineering any new features, three variables were hypothesised to matter most, based on domain knowledge of
the Sydney market: <b>building area</b> (living space is the single strongest lever on price across almost every housing
market), <b>suburb/location</b> (captured implicitly through distance to CBD and directly through the categorical suburb
feature — location premium dominates Sydney real estate), and <b>bedroom/bathroom count</b> (a standard proxy for
livable space and family suitability). The following engineered features were then created: <i>TotalRooms</i>
(bedrooms+bathrooms), <i>BedBathRatio</i>, <i>LandPerBed</i>, log-transformed distance to CBD and station (to linearise
diminishing marginal effects of distance), an <i>IsStrata</i> flag and <i>BldgImputed</i> flag for missing-data provenance,
and two NLP-derived scores from the agent listing text — <i>NLP_LuxuryScore</i> (counts of words such as "waterfront",
"panoramic", "designer", "prestigious") and <i>NLP_DevScore</i> (counts of words such as "potential", "renovate",
"subdivision") — plus a raw <i>WordCount</i>.""")
IMG('fig4_price_vs_area.png', caption="Figure 4. Price rises steeply and near-linearly with both land and building area, "
    "with Mosman properties commanding a visibly higher price per square metre than the other two suburbs at every size.")
IMG('fig5_price_vs_distcbd.png', caption="Figure 5. Price falls with distance to the CBD within each suburb, but the "
    "effect is dominated by the suburb-level jump — Blacktown's proximity variation barely moves price given its distance band.")
H2("2.4 Correlation Analysis: Did the Hypothesis Hold?")
IMG('fig6_corr_heatmap.png', width=13*cm, caption="Figure 6. Correlation matrix of numeric features against sale price.")
P("""The measured correlations with log(sale price) were: building area (r=0.87), total rooms (r=0.73), bathrooms
(r=0.70), bedrooms (r=0.66), word count (r=0.62), luxury score (r=0.61), distance to CBD (r=−0.56), and land area
(r=0.51). This <b>largely confirms the initial hypothesis</b>: building area was indeed the single strongest predictor,
and room counts/bathrooms were close behind. The main surprise was that the engineered <i>NLP luxury score</i> turned out
to be nearly as predictive as land area and bedrooms — marketing language is a meaningfully informative proxy for
unobserved quality (finishes, views, aspect) that structured fields do not capture — while <i>distance to station</i> and
<i>bed/bath ratio</i> contributed almost nothing (|r|&lt;0.2), suggesting they are largely redundant once suburb and
distance-to-CBD are already in the model.""")
story.append(PageBreak())

# ---------------- PART 3 ----------------
H1("Part 3 — Model Development and Evaluation")
H2("3.1 Model Selection and Pre-Training Expectations")
P("""Three models representing distinct modelling paradigms were selected. <b>Ridge Regression</b> is a regularised
linear model — fast, interpretable, and well-suited to small, noisy datasets with correlated features, but limited to
additive linear relationships (after the log/feature transforms). <b>Random Forest</b> is a bagged ensemble of decision
trees, capable of capturing non-linearities and interactions automatically, generally robust to outliers, but prone to
overfitting on small samples because each tree can memorise idiosyncratic training rows. <b>HistGradientBoosting</b> is a
boosted-tree ensemble that typically achieves the strongest accuracy on structured/tabular data at moderate-to-large
sample sizes, but is also the most flexible (highest-variance) of the three and therefore the most data-hungry.
<b>Pre-training expectation:</b> given only 87 training rows (after an 80/20 split) and 15+ features, the tree ensembles
were expected to outperform Ridge on training data but risk noticeably overfitting, while Ridge's regularisation was
expected to generalise more reliably despite being the "simplest" model on paper.""")

H2("3.2 Cross-Validated Results")
cv = pd.read_csv(D+'cv_results.csv')
cv_disp = cv.copy()
for c in ['Train R2','Val R2']: cv_disp[c] = cv_disp[c].round(3)
cv_disp['Val R2 Std'] = cv_disp['Val R2 Std'].round(3)
for c in ['Train RMSE ($)','Val RMSE ($)','Train MAE ($)','Val MAE ($)']: cv_disp[c] = cv_disp[c].round(0).astype(int).map('${:,}'.format)
for c in ['Train MAPE (%)','Val MAPE (%)']: cv_disp[c] = cv_disp[c].round(1)
story.append(make_table(cv_disp, fontsize=7))
story.append(Spacer(1,10))
IMG('fig7_cv_r2.png', width=11*cm, caption="Figure 7. Train vs 5-fold cross-validated R² per model (error bars = std across folds).")
IMG('fig8_cv_mape.png', width=11*cm, caption="Figure 8. Cross-validated MAPE per model.")

H2("3.3 Critical Analysis: Overfitting, Underfitting, and Complexity")
P("""The results confirm the pre-training expectation, and confirm it more strongly than anticipated. <b>Ridge Regression</b>
achieved the best validation R² (0.79) with the smallest train–validation gap (0.91 → 0.79), indicating good
generalisation and only mild overfitting. <b>Random Forest</b> achieved the highest training R² (0.94) but the lowest
validation R² (0.70) and the largest train–val gap — classic overfitting, where the ensemble is partly memorising the 87
training rows rather than learning generalisable structure. <b>HistGradientBoosting</b> sat in between (train R²=0.91,
val R²=0.72), also overfitting but less severely than Random Forest. Validation R² standard deviation across folds was
notably high for the tree models (0.24 and 0.22 vs Ridge's 0.17), showing their performance is unstable across different
train/validation splits — a direct symptom of too little data for their flexibility. No model showed clear
<i>underfitting</i> (all training R² values exceed 0.9); the dominant failure mode here is <b>excess model complexity
relative to sample size</b>, not insufficient capacity.""")

H2("3.4 Recommendation")
P("""<b>Ridge Regression is recommended</b> as the production model. It has the best and most stable validation
performance (R²=0.79, MAPE≈16.2%), the smallest overfitting gap, and — as a linear model on engineered, standardised
features — remains interpretable to a non-technical real-estate audience (coefficients map directly onto price-per-extra-
bedroom, price-per-sqm, etc.). This recommendation would very likely change with more data: both tree ensembles show
higher training capacity and would be expected to close or overtake Ridge's validation lead once sample size grows well
beyond ~100 properties, which is a concrete, testable prediction for future work.""")
story.append(PageBreak())

# ---------------- PART 4 ----------------
H1("Part 4 — Investigating Prediction Failures")
top5 = pd.read_csv(D+'top5_errors.csv')
disp = top5[['PropertyID','Suburb','PropertyType','SalePrice','Predicted','AbsError%']].copy()
disp['SalePrice'] = disp['SalePrice'].map('${:,.0f}'.format)
disp['Predicted'] = disp['Predicted'].round(0).astype(int).map('${:,}'.format)
disp['AbsError%'] = disp['AbsError%'].round(1)
story.append(make_table(disp, fontsize=8))
story.append(Spacer(1,10))
IMG('fig9_pred_vs_actual.png', width=10.5*cm, caption="Figure 9. Predicted vs actual price on the held-out test set (Ridge model); "
    "the five largest errors are highlighted in red.")
P("""The largest error (28.0%) was a Mosman whole-floor penthouse described as having "world-class 270-degree harbour
panoramas" — the model <b>under-predicted</b> by $4.19M. This is a ceiling effect: no comparable property in the small
Mosman training sample reaches this combination of apartment-format plus extreme, once-in-a-suburb view quality, so the
model regresses toward the suburb's more typical (still high, but lower) price level. Conversely, a $3.1M Marrickville
Victorian house was <b>over-predicted</b> by $869k (28.0% error) despite heritage-style marketing language ("ornate
ceilings," "double-fronted") — heritage character does not reliably command the same premium the luxury-word features
assume, particularly if the description omits renovation/condition signals the model can't see. A further two of the
top five errors are also Mosman houses with strong harbour-view language, and the fifth is an entry-level Mosman
apartment where the model over-predicted, suggesting the model has learned "Mosman + view words → high price" as a
strong heuristic that fails at the ends of that suburb's own price range (very top and very bottom).""")
P("""<b>Limitation implied by these cases:</b> the model has no access to genuinely differentiating information that a
human valuer would use — actual photographs, precise view corridor/aspect, interior finish quality, heritage listing
status, or floorplan efficiency. All five largest errors involve properties at the <i>extremes</i> of their suburb's
price distribution, supporting the general conclusion that <b>unique, high-value, or unusually characterful properties
are inherently harder to model</b> than "typical" mid-market stock, because comparable training examples are, by
definition, scarce. Predictions for such properties should be treated as a rough starting point rather than a reliable
estimate, and flagged for human review in any deployed system.""")
story.append(PageBreak())

# ---------------- PART 5 ----------------
H1("Part 5 — Human Judgement, Machine Learning, and Large Language Models")
P("""Ten properties were sampled from the held-out test set, spanning all three suburbs and both houses and apartments.
Three valuations were produced for each: (1) the best-performing ML model (Ridge); (2) an independent LLM valuation
produced by reasoning over the property's structured features and agent description using general Sydney market
knowledge (Claude, this report's author-assistant), <i>without</i> reference to the ML model's output; and (3) a
simplified "human-style" comparative market analysis (CMA) — the classic real-estate-agent heuristic of applying the
median $/sqm-of-building-area rate observed in comparable training-set sales (same suburb + property type) to each
property's building area. This CMA heuristic stands in for a human valuer's typical process; it is not an actual licensed
appraisal and this substitution is itself a limitation, discussed below.""")
p5 = pd.read_csv(D+'part5_final.csv')
disp5 = p5[['PropertyID','Suburb','SalePrice','ML_Pred','ML_Error','LLM_Pred','LLM_Error','HumanEst','Human_Error']].copy()
disp5.columns = ['ID','Suburb','Actual','ML $','ML %err','LLM $','LLM %err','Human $','Human %err']
for c in ['Actual','ML $','LLM $','Human $']: disp5[c] = disp5[c].round(0).astype(int).map('${:,}'.format)
for c in ['ML %err','LLM %err','Human %err']: disp5[c] = disp5[c].round(1)
story.append(make_table(disp5, fontsize=6.6))
story.append(Spacer(1,10))
bench = pd.read_csv(D+'part5_benchmark_metrics.csv')
bdisp = bench.copy()
bdisp['MAE ($)'] = bdisp['MAE ($)'].round(0).astype(int).map('${:,}'.format)
bdisp['RMSE ($)'] = bdisp['RMSE ($)'].round(0).astype(int).map('${:,}'.format)
bdisp['MAPE (%)'] = bdisp['MAPE (%)'].round(2)
bdisp['Correlation (r)'] = bdisp['Correlation (r)'].round(3)
story.append(make_table(bdisp, fontsize=8))
story.append(Spacer(1,10))
IMG('fig10_part5_benchmark.png', caption="Figure 10. Left: per-property absolute error by approach. Right: overall MAPE by approach.")
P("""On this small sample, the <b>CMA heuristic achieved the lowest overall error</b> (MAPE 9.5%, MAE $169k), narrowly
ahead of the ML model (MAPE 10.5%, MAE $246k), with the LLM valuation trailing (MAPE 12.9%, MAE $283k) — though all
three were highly correlated with actual prices (r&gt;0.99). This is a genuinely informative result: a simple,
suburb-specific $/sqm rate is a strong baseline precisely because location and building size are the dominant price
drivers in this market (Section 2.4), and a rate-based heuristic exploits that directly without needing to learn
interactions from a small sample. The <b>LLM valuation was weakest on outer-suburb properties with renovation language</b>
("potential to capitalise," MRK_007: 22.4% error) — plausibly because general knowledge about "renovator" language leads
to under-pricing relative to what has actually recently traded in a rapidly gentrifying pocket like Marrickville, a
local, recent-comparable-sales effect that neither general LLM knowledge nor a small ML training set fully captures.
The ML model was most accurate on well-represented, "typical" properties and, like in Part 4, weakest on Mosman
mid-to-upper properties where the local sample of true comparables is thin.""")
P("""<b>Does human judgement still add value?</b> Yes, but conditionally. The CMA heuristic's strength here comes
specifically from using recent local comparable sales — the same underlying advantage a human agent has from
neighbourhood familiarity and access to just-settled off-market data an ML model or LLM cannot see. Its weakness is that
it is a single linear rate and cannot flexibly combine multiple features the way Ridge does, nor draw on broad market
narrative the way an LLM can. In practice, the strongest real-world approach is likely a hybrid: an ML model for
consistent, feature-driven baseline pricing, cross-checked against recent local comparable rates (the human/CMA layer)
for properties near the edges of the training distribution — precisely the outlier cases identified in Part 4.""")
story.append(PageBreak())

# ---------------- PART 6 ----------------
H1("Part 6 — Final Deployment and Reflection")
H2("6.1 Application Development")
P("""A prototype web application was built with <b>Flask</b>, wrapping the trained Ridge Regression pipeline
(preprocessing + model, serialised with joblib). The user enters suburb, property type, bedrooms, bathrooms, car spaces,
land/building area, distance to CBD/station, and an optional listing description; the app derives the same engineered
features used in training (log-distances, total rooms, bed/bath ratio, land-per-bed, NLP luxury/development scores via
keyword counting, strata flag) and returns a predicted sale price in real time.""")
H2("6.2 How to Build, Run, and Use the Application")
P("""<b>Requirements:</b> Python 3.10+, <code>pip install flask joblib scikit-learn pandas numpy</code>.
<b>To run:</b> place <code>app.py</code>, <code>ridge_model.joblib</code>, and the <code>templates/index.html</code> file
in the same folder structure, then run <code>python app.py</code> and open <code>http://127.0.0.1:5000</code> in a
browser. <b>To use:</b> fill in the property form fields and click "Predict Sale Price"; the predicted price appears
immediately below the form.""")
IMG('app_screenshot_1_empty.png', width=10*cm, caption="Figure 11. The application's input form (screenshot of the running app).")
IMG('app_screenshot_2_result.png', width=10*cm, caption="Figure 12. A live prediction for a 4-bed/3-bath Mosman house with harbourside "
    "description: the deployed model returns $6,445,119.")

H2("6.3 Critical Reflection")
P("""This project's most important lesson was methodological rather than technical: the largest threat to a valid
property-valuation pipeline is not model choice but <b>data provenance and sample size</b>. With genuinely collected,
larger data, the tree ensembles (Random Forest, HistGBM) would likely be preferred over Ridge, reversing this report's
Part 3 recommendation — the current result is a direct artefact of having only 87 training rows. The feature-engineering
stage was the most defensible part of the pipeline: building area, suburb/location, and room counts dominated as
predicted, and the NLP luxury-score feature added genuine, non-trivial signal from unstructured text — a promising and
under-used data source in most conventional valuation tools.""")
P("""Ethically, an automated valuation tool trained on a specific, narrow set of suburbs carries clear fairness risks if
deployed more broadly: it will silently extrapolate outside its training distribution (as seen acutely in Part 4 for
extreme-value Mosman properties), it encodes whatever biases exist in its training sample (e.g., if certain suburbs or
property types are historically under-represented in sold-listing data, the model will be systematically less reliable
for them), and NLP features derived from agent marketing language risk indirectly encoding socioeconomic or even
demographic proxies embedded in real-estate marketing conventions. Any real deployment should report a confidence
interval or flag low-confidence predictions (e.g., properties far from the training distribution, as in Part 4) rather
than presenting a single point estimate as authoritative, and should be paired with human review for atypical
properties.""")
P("""With more time and resources, the highest-value improvements would be: (1) replacing the placeholder data with a
genuinely collected, substantially larger dataset (500+ properties per suburb) to test whether tree ensembles overtake
Ridge as hypothesised; (2) adding true image-based features (street view, floor plan) rather than relying on text proxies
for view/quality; (3) incorporating recent comparable sales as a dynamic, updating feature rather than a static training
set; and (4) building out proper model monitoring and recalibration as the Sydney market moves over time, since a
model trained on one period's settlement prices will drift as the market changes.""")

doc = SimpleDocTemplate('/home/claude/project/report/Sydney_Housing_Valuation_Report.pdf', pagesize=A4,
                         topMargin=1.8*cm, bottomMargin=1.8*cm, leftMargin=2*cm, rightMargin=2*cm,
                         title="Sydney Housing Price Prediction and Decision Support System")
doc.build(story)
print("PDF built.")

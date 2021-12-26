* ------------------------------------------------------------------------------
* 
* Estimating the model for Karp 2019 (working) -- "Gains from trade in 
* emissions permits"
*
* See Appendix B for details (3/5/20 version of the paper).
*
* A. Hultgren, hultgen@berkeley.edu
* 1/6/20
*
* ------------------------------------------------------------------------------


clear all

cd "\\tsclient\E\Documents\kcgcGoogleDrive\College_future\PhD\research\Larry\emissions estimation\summary\Fr_Andy_May11"

* Data cleaning was done in R, just import the .csv

import delimited data\clean\ts_allYears_nation.1751_2014.csv
replace time_lag = "." if time_lag == "NA"
replace ebar_lag = "." if ebar_lag == "NA"
destring, replace
drop v1

gen time2 = time^2
gen time2_lag = time_lag^2

cd replication

* Estimate Eq. 19 with nonlinear regression. 

local min_year 1945
local max_years 2010 2005
eststo clear
local titles ""


// local max_year 2005
// nl (ebar = {p=0.1}*ebar_lag + {B0-pB0=1} + {t1=1}*time + {t2=1}*time2 + ///
// 	{t1}*{p}*time_lag + {t2}*{p}*time2_lag) ///
// 	if (year >= `min_year') & (year <= `max_year'), ///
// 	variables(ebar_lag time time_lag time2 time2_lag) ///
// 	title("Estimates of Equation 19")


foreach yrmax of local max_years {
	
	nl (ebar = {p=0.1}*ebar_lag + {B0-pB0=1} + {t1=1}*time + {t2=1}*time2 + ///
	{t1}*{p}*time_lag + {t2}*{p}*time2_lag) ///
	if (year >= `min_year') & (year <= `yrmax'), ///
	variables(ebar_lag time time_lag time2 time2_lag) ///
	title("Estimates of Equation 19")
	eststo model`yrmax'
	local titles `" 	`titles' "`min_year'-`yrmax'"	"'
	estimates save ster/eq19_`yrmax', replace
	estimates store eq19_`yrmax'
	
	
	* Solve for the estimate of B0.
	nlcom _b[/B0-pB0]/(1-_b[/p])
	matrix B0 = r(b)
	global B0_`yrmax' = B0[1,1]
	matrix vcv = r(V)
	global se_B0_`yrmax' = (vcv[1,1])^0.5
	
	* Calculate sigma^2 / b^2 in the paper (note the B1^2 will drop out when
	* calculating the ratio k).
	predict res_`yrmax', residuals
	gen res2_`yrmax' = (res_`yrmax')^2
	sum res2_`yrmax' if ((year >= `min_year') & (year <= `yrmax'))
	local mySSR = r(sum)
	di("SSR:")
	di(`mySSR')
	local denom = (`yrmax' - `min_year' - 3 - 2)
	global sigma2_`yrmax' = `mySSR' / (`yrmax' - `min_year' - 3 - 2)
	display("Estimate of sigma^2 / B1^2 for ending year `yrmax': ${sigma2_`yrmax'}")
	
	
	* Estimate the standard error of sigma^2 / b^2, following Eq (3) in 
	* https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
	global se_sigma2_`yrmax' = (${sigma2_`yrmax'}^0.5) * ( 1 / (2*(`yrmax' - `min_year' - 3 - 2)) )^0.5
	
}

// Try with correct signs in the model
// foreach yrmax of local max_years {
//	
// 	nl (ebar = {p=0.1}*ebar_lag + {B0-pB0=1} + {t1=1}*time + {t2=1}*time2 - ///
// 	{t1}*{p}*time_lag - {t2}*{p}*time2_lag) ///
// 	if (year >= `min_year') & (year <= `yrmax'), ///
// 	variables(ebar_lag time time_lag time2 time2_lag) ///
// 	title("Estimates of Equation 19 (Correct Signs)")
// 	eststo model`yrmax'_signs
// 	local titles `" 	`titles' "`min_year'-`yrmax'"	"'
// 	estimates save ster/eq19_`yrmax'_signs, replace
// 	estimates store eq19_`yrmax'_signs
//	
//	
// 	* Solve for the estimate of B0.
// 	nlcom _b[/B0-pB0]/(1-_b[/p])
// 	matrix B0 = r(b)
// 	global B0_`yrmax' = B0[1,1]
// 	matrix vcv = r(V)
// 	global se_B0_`yrmax' = (vcv[1,1])^0.5
//	
// 	* Calculate sigma^2 / b^2 in the paper (note the B1^2 will drop out when
// 	* calculating the ratio k).
// 	predict res_`yrmax'_signs, residuals
// 	gen res2_`yrmax'_signs = (res_`yrmax')^2
// 	sum res2_`yrmax'_signs if ((year >= `min_year') & (year <= `yrmax'))
// 	local mySSR = r(sum)
// 	local denom = (`yrmax' - `min_year' - 3 - 2)
// 	global sigma2_`yrmax'_signs = `mySSR' / (`yrmax' - `min_year' - 3 - 2)
// 	display("Estimate of sigma^2 / B1^2 for ending year `yrmax' (corrected signs): ${sigma2_`yrmax'_signs}")
//	
//	
// 	* Estimate the standard error of sigma^2 / b^2, following Eq (3) in 
// 	* https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
// 	global se_sigma2_`yrmax'_signs = (${sigma2_`yrmax'}^0.5) * ( 1 / (2*(`yrmax' - `min_year' - 3 - 2)) )^0.5
//	
// }

// mac li
esttab model* using output/eq19.doc, se mtitle(`titles') replace









* Estimate Eq. 20, where group fixed effects estimate (b_0i - B0)/b.

* Load the data and create group fixed effects
clear
eststo clear
clear mata
local titles ""
local min_year 1945
local max_years 2010 2005
local yrmax 2010

foreach yrmax of local max_years {

	clear
	
	display "`yrmax'"

	import delimited data\clean\grouped_allYears_nation.1751_2014.csv
	drop v1

	levelsof group, local(grouplevels)
	foreach gp of local grouplevels {
		
		capture drop b_0`gp'_B0
		gen b_0`gp'_B0 = 0
		replace b_0`gp'_B0 = 1 if group=="`gp'"
	}

	* Calculate average emissions in each year, and subtract from each region's emissions.
	bysort year: egen ebar = mean(co2)
	gen co2_demeaned = co2 - ebar

	* This ordering is necessary to make the matrix math work out for GLS below
	sort year group

	* Calculate weighted y and X for GLS
	* Helpful reference: https://blog.stata.com/2015/12/15/programming-an-estimation-command-in-stata-mata-101/
	* and https://www.schmidheiny.name/teaching/statamata.pdf

	preserve

	keep if (year >= `min_year') & (year < `yrmax') & (group != "USA")
	drop b_0USA_B0
	
	tab group

	mata {

		V_inv = (1/3 * sqrt(3), 0 			 , 0 \ ///
				1/12 * sqrt(6), 1/4 * sqrt(6), 0 \ ///
				1/4 * sqrt(2) , 1/4 * sqrt(2), 1/2 * sqrt(2));

		V_inv;
		
		T = strtoreal(st_local("yrmax")) - strtoreal(st_local("min_year"));
		
		T;
		
		I_T = I(T);
		
		W = I_T # V_inv;
		
		W[| 1,1 \ 4,4 |];
		
		y = st_data(., "co2_demeaned");
		y[| 1 \ 4 |];
		
		y_weighted = W * y;
		y_weighted[| 1 \ 4 |];
		
		st_addvar("float", ("co2_weighted"));
		st_store(., ("co2_weighted"), y_weighted);
		
		X = st_data(., "b_*");
		X[| 1,1 \ 6,3 |];
		
		// X
		
		X_weighted = W * X;
		
		X_weighted[| 1,1 \ 6,3 |];
		
		new_Xs = st_addvar("float", ("b0BRIC_weighted", "b0EU_weighted", "b0Other_weighted"));
		st_store(., new_Xs, X_weighted);

	};
	
	list in 1/5

	* GLS regression

	reg co2_weighted b0*_w*, noconstant
	
	matrix results = r(table)
	matrix list results
	eststo model`yrmax'
	local titles `" 	`titles' "`min_year'-`yrmax'"	"'
	estimates save ster/eq20_`yrmax', replace
	estimates store eq20_`yrmax'


	capture drop resid resid2
	predict resid, residuals
	gen resid2 = resid^2
	sum(resid2)
	local ssr = r(sum)

	di("`ssr'")

	global sigma2_u_`yrmax' = `ssr' / (3 * (`yrmax' - `min_year') - 4)

	display("SSR/((n-1)T-n): ${sigma2_u_`yrmax'}")
	
	
	
	
	local myN = 4
	local sigmas_ratio = (`myN'-1) * ${sigma2_u_`yrmax'} / ${sigma2_`yrmax'}
	
	display(" --------------------------- ")
	display(" Years `min_year' to `yrmax' ")
	display("(n-1)/n * sigma2_u / sigma2: ")
	display( `sigmas_ratio' )
	display(${sigma2_u_`yrmax'}) 
	display(${sigma2_`yrmax'})
	display(" --------------------------- ")
	
	* From Eq. 9, calculate the last (b_0j - B0)/b.  (Here, j = USA). 
	* Note nB0 = sum(b_0i) over all i, so b_0j - B0 = -(sum(b_0i-B0)) for i != j,
	* so (b_0j - B0)/b = -(sum(b_0i-B0)/b ) for i != j
	local myN_1 = `myN'-1
	local sum_otherFEs = 0
	local sum_b0s = 0
	local sum_squares_b0s = 0
	forvalues i = 1/`myN_1' {
		local sum_otherFEs = `sum_otherFEs' + results[1,`i']
		local tmp_sum = results[1,`i'] + ${B0_`yrmax'}
		local sum_b0s = `sum_b0s' + `tmp_sum'
		local sum_squares_b0s = `sum_squares_b0s' + (`tmp_sum')^2
	}
	global b_0USA_B0_`yrmax' = 0 - `sum_otherFEs'
	
	* Calculate var(b_0) from Eq. 7
	local sum_b0s = `sum_b0s' + ${b_0USA_B0_`yrmax'} + ${B0_`yrmax'}
	local sum_squares_b0s = `sum_squares_b0s' + (${b_0USA_B0_`yrmax'} + ${B0_`yrmax'})^2
	global var_b0_`yrmax' = 1/`myN' * `sum_squares_b0s' - (1/`myN' * `sum_b0s')^2
	
	
	* Write all calculated values to a .csv file
	file open myfile using "output\calculated_values_`yrmax'.csv", write replace
	file write myfile ///
			"variable,estimate,s.e.,sample" _n ///
			"ratio k,`sigmas_ratio',NA,`min_year'-`yrmax'" _n ///
			"sigma2 / b^2,${sigma2_`yrmax'},${se_sigma2_`yrmax'},`min_year'-`yrmax'" _n ///
			"sigm2_u / nb^2,${sigma2_u_`yrmax'},${se_sigma2_u_`yrmax'},`min_year'-`yrmax'" _n ///
			"B0,${B0_`yrmax'},${se_B0_`yrmax'},`min_year'-`yrmax'" _n ///
			"var_b0,${var_b0_`yrmax'},NA,`min_year'-`yrmax'" _n ///
			"(b_0USA-B0)/b,${b_0USA_B0_`yrmax'},NA,`min_year'-`yrmax'"
	file close myfile
	
	
	


	* As a check against the above estimate, estimate Eq. 25 (March 5 draft).

	mata {


		T = strtoreal(st_local("yrmax")) - strtoreal(st_local("min_year"));
		
		T;
		
		I_T = I(T);
		
		s = J(T, 1, 1);
		
		// s
		
		Omega = ( 3, -1, -1 \ ///
				 -1,  3, -1 \ ///
				 -1, -1,  3 );
		
		Omega;
		
		n = 4;
		
		y = st_data(., "co2_demeaned");
		
		// y
		
		eq25 = y' * ( (I_T - (1/T)*s*s') # cholinv(Omega) ) * y / ((n-1)*T-n);

		eq25;
		
		// det(Omega)

	};

	display("SSR/((n-1)T-n): ${sigma2_u_`yrmax'}")
	* eq25 output matches the GLS estimator of the variance to the first decimal place. Great!

	restore

}

esttab model* using output/eq20.doc, se mtitle(`titles') replace



#' Inflated distributional regression
#' 
#' Fit a value(s)-inflated regression, potentially with
#' additive predictors driving the inflation part(s) and
#' different additive predictors for the actual distribution
#'
#' @param y response
#' @param family character specifying the actual distribution that is 
#' zero-inflated; see \code{?deepregression} for possible options
#' @param list_of_formulas a list of formulas of length 
#' \code{nr_dist_param} for all additive predictors of the 
#' actual distribution part
#' @param formula_inflation formula for the additive predictor 
#' that influences the probability of inflation(s); default
#' is \code{~ 1} which learns one covariate-independent parameter
#' for the probability of (each) inflation
#' @param list_of_deep_models see \code{?deepregression}
#' @param data data.frame or list with data
#' @param ... further arguments passed to \code{?deepregression}
#' 
#' @return a model of class \code{mixdistreg}, \code{inflareg} and
#' \code{deepregression}
#'
#' @rdname inflareg
#'
#' @export
#' @import deepregression
#' 
#' @examples 
#' n <- 1000
#' data = data.frame(matrix(rnorm(4*n), c(n,4)))
#' colnames(data) <- c("x1","x2","x3","xa")
#' formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
#'
#' deep_model <- function(x) x %>%
#' layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
#' layer_dropout(rate = 0.2) %>%
#' layer_dense(units = 8, activation = "relu") %>%
#' layer_dense(units = 1, activation = "linear")
#'
#' y <- rnorm(n) + data$xa^2 + data$x1
#'
#' mod <- inflareg(
#'   list_of_formulas = list(loc = formula, scale = ~ 1),
#'   formula_inflation = ~ 1 + x1,
#'   data = data, 
#'   y = y,
#'   list_of_deep_models = list(deep_model = deep_model),
#'   inflation_values = 1.0
#' )
#'
#' if(!is.null(mod)){
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' 
#' }
#'
#' mod <- inflareg(
#'   list_of_formulas = list(loc = formula, scale = ~ 1),
#'   formula_inflation = ~ 1 + x1,
#'   data = data, 
#'   y = y,
#'   list_of_deep_models = list(deep_model = deep_model),
#'   inflation_values = c(0.0,1.0)
#' )
#'
#' if(!is.null(mod)){
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' 
#' }
inflareg <- function(
    y,
    family = "normal",
    list_of_formulas,
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values,
    data,
    ...
)
{
  
  mod <- mixdistreg(y = y,
             families = c(family, rep("deterministic", 
                                      length(inflation_values))),
             type = "general",
             nr_comps = 1L + length(inflation_values),
             list_of_formulas = list_of_formulas,
             formula_mixture = formula_inflation,
             list_of_deep_models = list_of_deep_models,
             inflation_values = inflation_values,
             data = data,
             ...
  )
  
  class(mod) <- c(class(mod), "inflareg")
  
  return(mod)
  
}

#' Zero-inflated distributional regression
#' 
#' Fit a zero-inflated regression, potentially with
#' additive predictors driving the zero-inflated part and
#' different additive predictors for the actual distribution
#'
#' @param ... see \code{?inflareg}
#'
#' @rdname inflareg
#'
#' @export
zinreg <- function(
    ...
){
  
  inflareg(..., inflation_values = 0.0)
  
}

#' One-inflated distributional regression
#' 
#' Fit a one-inflated regression, potentially with
#' additive predictors driving the one-inflated part and
#' different additive predictors for the actual distribution
#' 
#' @param ... see \code{?inflareg}
#'
#' @rdname inflareg
#'
#' @export
oinreg <- function(
    ...
){
  
  inflareg(..., inflation_values = 1.0)
  
}

#' Zero-one-inflated distributional regression
#' 
#' Fit a zero-one-inflated regression, potentially with
#' additive predictors driving the zero- and one-inflated part and
#' different additive predictors for the actual distribution
#' 
#' @param ... see \code{?inflareg}
#'
#' @rdname inflareg
#'
#' @export
zoinreg <- function(
    ...
){

  inflareg(..., inflation_values = c(0.0, 1.0))  
  
}



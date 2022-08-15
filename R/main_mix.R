#' Generic mixture distributional regression
#' 
#' Fit a mixture distributional regression 
#'
#' @param y response
#' @param families character (vector); 
#' see the \code{family} argument of \code{deepregression}. Can be multiple
#' distributions for a mixture of different distributions (\code{type = "general"})
#' @param nr_comps integer; number of mixture components
#' @param list_of_formulas a list of formulas for the 
#' parameters of the distribution(s) which are used in the mixture
#' (i.e., for \code{family="normal"} a list of two formulas
#' is required). The elements in the list first consist of all parameters for
#' the first distribution, then all for the second distribution and so on.
#' See also \code{?deepregression} for specification details for the formulas.
#' @param formula_mixture formula for the the additive predictor 
#' of the mixture component. Covariate-independent per default.
#' @param list_of_deep_models see \code{?deepregression}
#' @param data data.frame or list with data
#' @param ... further arguments passed to \code{?deepregression}
#' 
#' @return a model of class \code{mixdistreg} and
#' \code{deepregression}
#' 
#' @examples 
#' 
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
#' mod <- mixdistreg(
#'   families = c("normal", "student_t"),
#'   type = "general",
#'   list_of_formulas = list(
#'     loc = formula, scale = ~ 1, 
#'     df = formula
#'     ),
#'    
#'   formula_mixture = ~ 1 + x1,
#'   data = data, 
#'   y = y,
#'   list_of_deep_models = list(deep_model = deep_model),
#'   inflation_values = NULL,
#'   optimizer = optimizer_adam(learning_rate=1e-6)
#' )
#'
#' if(!is.null(mod)){
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' 
#' }
#' 
#'
#' @export
#' @import deepregression
mixdistreg <- function(
    y,
    families = c("normal"),
    type = c("same", "general"),
    nr_comps = 2L,
    list_of_formulas,
    formula_mixture = ~1,
    list_of_deep_models = NULL,
    inflation_values = NULL,
    data,
    trafos_each_param = NULL,
    ...
)
{
  
  type <- match.arg(type)

  # create family
  dist_fun <- gen_mix_dist_maker(
    type = type,
    families = families,
    nr_distributions = nr_comps,
    trafos_each_param = trafos_each_param,
    inflation_values = inflation_values
  )
  
  tep <- get("trafos_each_param", environment(dist_fun))
  
  if(length(list_of_formulas) != length(unlist(tep)))
    stop("List of formulas (", length(list_of_formulas), 
         ") does not match the number of predictors (",
         length(unlist(tep)),
         ")")
  
  list_of_formulas <- 
    c(list(mixture = formula_mixture),
      list_of_formulas)
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list_of_formulas,
    list_of_deep_models = list_of_deep_models,
    family = dist_fun,
    # from_distfun_to_dist = distfun_to_dist_mix(nr_comps,
    #                                            length(list_of_formulas)),
    output_dim = c(nr_comps, rep(1L, length(list_of_formulas)-1L)),
    ...
  )
  
  class(mod) <- c("mixdistreg", class(mod))
  
  mod$init_params$mixture_specification <- 
    list(
      families = families,
      type = type,
      inflation_values = inflation_values,
      trafos_each_param = tep
    )
  
  return(mod)
  
}

#' Same-same mixture distributional regression
#' 
#' Fit a mixture distributional regression with same family mixture
#' and same predictors for each mixture
#'
#' @param y response
#' @param family character; see \code{?deepregression}
#' @param nr_comps integer; number of mixture components
#' @param list_of_formulas a list of formulas for the 
#' parameters of the distribution which is used in the mixture
#' (i.e., for \code{family="normal"} only a list of two formulas
#' is required). See also \code{?deepregression} for more details.
#' @param formula_mixture formula for the the additive predictor 
#' of the mixture component. Covariate-independent per default.
#' @param list_of_deep_models see \code{?deepregression}
#' @param data data.frame or list with data
#' @param ... further arguments passed to \code{?deepregression}
#' 
#' @return a model of class \code{mixdistreg} and
#' \code{deepregression}
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
#' mod <- sammer(
#'   list_of_formulas = list(loc = formula, scale = ~ 1),
#'   nr_comps = 3,
#'   data = data, y = y,
#'   list_of_deep_models = list(deep_model = deep_model)
#' )
#'
#' if(!is.null(mod)){
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' 
#' }
#'
sammer <- function(
  y,
  family = "normal",
  nr_comps = 2L,
  list_of_formulas,
  formula_mixture = ~1,
  list_of_deep_models = NULL,
  data,
  ...
)
{
  
  # define list_of_formulas for deepregression
  org_len <- length(list_of_formulas) 
    
  if(org_len == 1)
    list_of_formulas <- list_of_formulas[rep(1, nr_comps)]
  
  list_of_formulas <- rep(list_of_formulas, nr_comps)
  
  names(list_of_formulas) <- paste0(rep(names_families(family), nr_comps),
                                    "_mix", rep(1:nr_comps, each=org_len))
  
  return(
    mixdistreg(y = y,
    families = family,
    type = "same",
    nr_comps = nr_comps,
    list_of_formulas = list_of_formulas,
    formula_mixture = formula_mixture,
    list_of_deep_models = list_of_deep_models,
    data = data,
    ...
    )
  )
  
}

# distfun_to_dist_mix <- function(nr_comps, nr_params){
#   function(dist_fun, preds)
#   {
# 
#     list_pred <- layer_lambda(preds,
#                               f = function(x)
#                               {
#                                 tf$split(x, num_or_size_splits =
#                                            c(nr_comps, as.integer(nr_params-1)),
#                                          axis = 1L)
#                               })
#     list_pred[[1]] <- list_pred[[1]] %>%
#       layer_activation(activation = "softmax")
#     # preds <- layer_concatenate(list_pred)
#     names(list_pred) <- c("probs", "params")
# 
#     return(distfun_to_dist(dist_fun, list_pred))
# 
#   }
# }
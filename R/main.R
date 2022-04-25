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
#' mod <- mixdistreg(
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
mixdistreg <- function(
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
  
  # create family
  dist_fun <- mix_dist_maker(
    nr_comps = nr_comps,
    dist = family_to_tfd(family),
    trafos_each_param = family_to_trafo(family)
  )
  
  # define list_of_formulas for deepregression
  org_len <- length(list_of_formulas) 
    
  if(length(list_of_formulas) == 1)
    list_of_formulas <- list_of_formulas[rep(1, nr_comps)]
  
  list_of_formulas <- 
    c(list(mixture = formula_mixture),
      rep(list_of_formulas, nr_comps))
  
  names(list_of_formulas)[-1] <- paste0(rep(names_families(family), nr_comps),
                                           "_mix", rep(1:nr_comps, each=org_len))
  
  
  
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulas = list_of_formulas,
    list_of_deep_models = list_of_deep_models,
    family = dist_fun,
    from_distfun_to_dist = distfun_to_dist_mix(nr_comps, length(list_of_formulas)),
    ...
  )
  
  class(mod) <- c("mixdistreg", class(mod))
  
  return(mod)
  
}


#' generate mixture distribution of same family
#'
#' @param dist tfp distribution
#' @param nr_comps number of mixture components
#' @param trafos_each_param list of transformaiton applied before plugging
#' the linear predictor into the parameters of the distributions.
#' Should be of length #parameters of \code{dist}
#' @return returns function than can be used as argument \code{dist\_fun} for
#' @export
#'
mix_dist_maker <- function(
  dist = tfd_normal,
  nr_comps = 3,
  trafos_each_param = list(function(x) x, function(x) tf$add(1e-8, tfe(x)))
  ){

  stack <- function(x,ind=1:nr_comps) tf$stack(
    lapply(ind, function(j)
      tf_stride_cols(x,j)), 2L)

  mixdist = function(probs, params)
  {

    mix = tfd_categorical(probs = probs)
    this_components = do.call(dist, params)

    res_dist <- tfd_mixture_same_family(
      mixture_distribution=mix,
      components_distribution=this_components
    )

    return(res_dist)
  }

  trafo_fun <- function(x){

    c(probs = list(stack(x,1:nr_comps)),
      params = list(
        lapply(1:length(trafos_each_param),
               function(i){
                 ind <- nr_comps +
                   # first x for pis
                   (i-1)*nr_comps +
                   # then for each parameter
                   (1:nr_comps)
                 stack(trafos_each_param[[i]](
                   tf_stride_cols(x,min(ind),max(ind))
                 )
                 )
               }
        )
      )
    )

  }

  return(
    function(x) do.call(mixdist, trafo_fun(x))
  )

}



distfun_to_dist_mix <- function(nr_comps, nr_params){
  function(dist_fun, preds)
  {
    
    list_pred <- layer_lambda(preds,
                              f = function(x)
                              {
                                tf$split(x, num_or_size_splits =
                                           c(1L, as.integer(nr_params-1)),
                                         axis = 1L)
                              })
    list_pred[[1]] <- list_pred[[1]] %>%
      layer_dense(units = as.integer(nr_comps),
                  activation = "softmax",
                  use_bias = FALSE)
    preds <- layer_concatenate(list_pred)
    
    return(distfun_to_dist(dist_fun, preds))
    
  }
}
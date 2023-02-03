#' Generic methods for mixdistreg objects
#' 
#' The main difference to methods from deepregression
#' is that these allow to define strings for \code{which_param}
#' to explicitly extract a coefficients, plots, ... for 
#' either the \code{"mixture"} or a defined distribution family.
#'
#' @param object mixdistreg object
#' @param which_dist integer or string. For integer, see
#' \code{?coef.deepregression}. If a string is provided, it
#' must be one of \code{"mixture"} or a family string used
#' to define the distributions in the mixture (e.g. \code{"normal"}).
#' Can also be a combination of the family and a number to specify which
#' of several distributions with the same family (e.g., \code{"normal2"}
#' for the second normal distribution) 
#' @param which_param see \code{?coef.deepregression}
#' @param type see \code{?coef.deepregression}
#' @param ... further arguments, passed to fit, plot or predict function
#'
#'
#' @method coef mixdistreg
#' @export
#' @rdname methodMix
#' 
coef.mixdistreg <- function(
    object,
    which_dist = "mixture",
    which_param = 1L,
    type = NULL,
    ...
    )
{
  
  class(object) <- class(object)[-1]
  
  if(which_dist == "mixture" | which_dist == 1)
    return(
      coef(object = object,
           which_param = 1L,
           type = type,
           ...)
    )
  
  # else: other dist param
  which_param <- match_which_param(
    families = object$init_params$mixture_specification$families,
    trafos = object$init_params$mixture_specification$trafos_each_param,
    which_dist,
    which_param
  )
  
  coef(object = object,
       which_param = which_param,
       type = type,
       ...)  
    
}

#' @method plot mixdistreg
#' @export
#' @rdname methodMix
#'
plot.mixdistreg <- function(
    x,
    which_dist = "mixture",
    which = NULL,
    which_param = 1,
    only_data = FALSE,
    grid_length = 40,
    main_multiple = NULL,
    type = "b",
    get_weight_fun = get_weight_by_name,
    ...
)
{
  
  class(x) <- class(x)[-1]
  
  if(which_dist == "mixture" | which_dist == 1)
    return(
      plot(
        x = x,
        which = which,
        which_param = 1L,
        only_data = only_data,
        grid_length = grid_length,
        main_multiple = main_multiple,
        type = type,
        get_weight_fun = get_weight_fun,
        ...
      )  
    )
  
  # else: other dist param
  which_param <- match_which_param(
    families = x$init_params$mixture_specification$families,
    trafos = x$init_params$mixture_specification$trafos_each_param,
    which_dist,
    which_param
  )
  
  plot(
    x = x,
    which = which,
    which_param = which_param,
    only_data = only_data,
    grid_length = grid_length,
    main_multiple = main_multiple,
    type = type,
    get_weight_fun = get_weight_fun,
    ...
  )  
  
}

match_which_param <- function(families, trafos, which_dist, which_param){
  
  if(grepl("[1-9]", which_dist) & is.character(which_dist))
  {
    
    nr <- as.numeric(gsub(".*([1-9])", "\\1", which_dist))
    which_dist <- gsub("[1-9]", "", which_dist)
    
  }else{
    
    nr <- NULL
    
  }
  
  if(length(families)==1) 
    families <- rep(families, length(trafos))
  possible_chars <- c("mixture", families)
  if(is.character(which_dist)){
    which_dist <- which(which_dist == possible_chars)
  }
  
  if(!is.null(nr)) which_dist <- which_dist[nr]
  
  sizes <- sapply(trafos, length)
  
  if(length(sizes)>2 & which_dist!=2)
    which_param <- sum(sizes[1:(which_dist-2)]) + which_param + 1L else
      which_param <- which_param + 1L
  
  return(which_param)
  
}

#' @param object mixdistreg object
#' @param convert_fun function; to convert Tensor
#' @param posterior logical; should the a posterior probabilities
#' be returned or the estimated values; default is FALSE
#' @param data data.frame or list; optional, providing new data
#' @param this_y vector; optional, new outcome corresponding to data
#' @export
#' @return a matrix with columns corresponding to clusters and values
#' in the matrix to probabilities
#' @rdname methodMix
#'
get_pis <- function(object, convert_fun=as.array, data=NULL, 
                    this_y=NULL, posterior=FALSE)
{
  
  dist_dr <- get_distribution(object, data=data)
  if(!posterior) return(convert_fun(dist_dr$mixture_distribution$probs)[,1,])
  
  if(posterior & !inherits(object, "sammer"))
    stop("A posterior probabilities not yet implented for non-same mixture models.")
  
  if(!is.null(data) && is.null(this_y)) stop("Must provide outcome this_y if data is provided.")
  outcome <- if(is.null(data)) object$init_params$y else this_y
  
  return(
    as.matrix(tf$squeeze(
    dist_dr$submodules[[1]]$prob(
      array(outcome, dim = c(NROW(outcome),1,1))), 
    axis=1L))
  )
  
}

#' @param object mixdistreg object
#' @param convert_fun function; to convert Tensor
#' @param data data.frame or list; optional, providing new data
#' @param what character; specifying what object from the fitted distribution
#' to return. Can be one of the following: \code{"means"} for the means of all
#' distributions, \code{"stddev"} for the standard deviation of all
#' distributions, \code{"quantile"} for the quantile which is provided in
#' the argument \code{value}, \code{"cdf"} for values of the
#' CDFs evaluated at \code{value}(s), \code{"prob"} for the
#' pdfs at \code{value}(s), \code{"loc"} for the distributions' location or
#' \code{"scale"} for the distributions' scale.
#' @param value numeric vector; value(s) provided for different functions specified 
#' in \code{what}
#' @export
#' @return a matrix 
#' @rdname methodMix
#'
get_stats_mixcomps <- function(object, convert_fun=as.matrix, 
                               data = NULL,
                               what, value = NULL)
{
  
  mixcomps <- get_distribution(object, data = data)$submodules[[1]]
  
  if(!inherits(object, "sammer"))
    stop("Function currently only implemented for same-mixture models.")

  if(length(value)>1){ 
    shape_dist <- mixcomps$batch_shape$as_list()
    if(length(value) != shape_dist[1]) stop("value must be either scalar or of size nrow(data)")
    # broadcast to correct shape
    value <- array(rep(value, shape_dist[3]), dim = shape_dist)
  }
  
  res <- switch(what,
         means = tf$squeeze(mixcomps$mean()),
         stddev = tf$squeeze(mixcomps$stddev()),
         quantile = tf$squeeze(mixcomps$quantile(value = value)),
         cdf = tf$squeeze(mixcomps$cdf(value = value)),
         prob = tf$squeeze(mixcomps$prob(value = value)),
         loc = tf$squeeze(mixcomps$loc()),
         scale = tf$squeeze(mixcomps$scale())
         )
  
  return(convert_fun(res))
  
}



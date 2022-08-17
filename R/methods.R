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
    families = object$init_params$mixture_specification$families,
    trafos = object$init_params$mixture_specification$trafos_each_param,
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

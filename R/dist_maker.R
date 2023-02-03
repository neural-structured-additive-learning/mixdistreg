#' Generates a general mixture distribution
#' 
#' This function supplies the functionality to define mixture distribution
#' from TFP, including the 
#' 
#' @param families character (vector); 
#' see the \code{family} argument of \code{deepregression}. Can be multiple
#' distributions for a mixture of different distributions
#' @param nr_distributions integer; defining the number of distributions
#' to be mixed (including deterministic ones)
#' @param trafos_each_param list of (lists of) transformations applied before plugging
#' the linear predictor into the parameters of the distributions.
#' Should be of length #parameters of \code{dist}; uses the defaults
#' from \code{deepregression} if NULL.
#' @param dist_fun tfd distribution defining the mixture
#' @examples 
#' 
#' library(deepregression)
#' 
#' # a function that can be passed to deepregression
#' # as family
#' gen_mix_dist_maker(
#'   type = "same",
#'   nr_distributions = 2L
#'  )
#'  
#' gen_mix_dist_maker(
#'   families = c("normal", "student_t"),
#'   trafos_each_param = 
#'   list(
#'        list(function(x) x, 
#'             function(x) tf$add(add_const, tfe(x))
#'        ),
#'   ),
#'   list(
#'        function(x) x, 
#'        function(x) tf$add(add_const, tfe(x))
#'        )
#'   ),
#'   type = "general",
#'   nr_distributions = 2L
#'  )
#' 
#' @export
#' @rdname dist_maker
#' 
gen_mix_dist_maker <- function(
    type = c("same", "general"),
    families = c("normal"),
    nr_distributions = 1L,
    trafos_each_param = NULL,
    inflation_values = NULL
){
  
  type <- match.arg(type)
  
  stack <- function(x,ind=1:nr_distributions) tf$stack(
    lapply(ind, function(j)
      tf_stride_cols(x,j)), 2L)
  
  if(is.null(trafos_each_param))
    trafos_each_param <- lapply(families[families!="deterministic"], 
                                       family_to_trafo)#,
                                # recursive = FALSE)
  if(type == "same"){
    trafos_each_param <- trafos_each_param[rep(1, nr_distributions)]
    families <- rep(families[1], nr_distributions)
  }
    
  dists <- lapply(families, family_to_tfd)
  
  if(length(trafos_each_param) != length(dists)-length(inflation_values))
    stop("Number of families does not match the number of transformations.")
  
  if(type == "same"){
    
    mixdist = function(probs, params)
    {
      
      mix = tfd_categorical(probs = probs)
      params = lapply(1:length(trafos_each_param[[1]]),
                      function(i){
                        ind <- seq(i, i+((nr_distributions-1)*length(trafos_each_param[[1]])),
                                   by = length(trafos_each_param[[1]]))
                        stack(
                          tf$keras$layers$Lambda(function(x) x[,ind])(params)
                          )
                      }
      )
      
      this_components = do.call(dists[[1]], params)
      
      res_dist <- do.call(tfd_mixture_same_family, args = list(mix, this_components))
      
      return(res_dist)
    }
    
  }else if(type == "general"){
    
    mixdist = function(probs, params)
    {
      
      mix = tfd_categorical(probs = probs)
      split_sizes <- sapply(trafos_each_param, length)
      if(!is.null(inflation_values))
        split_sizes <- c(split_sizes, rep(1, length(inflation_values)))
      params <- tf$split(params, num_or_size_splits = as.integer(split_sizes), axis = -1L)
      this_components = lapply(1:length(split_sizes), function(i) 
        do.call(dists[[i]], #lapply(
          tf$split(params[[i]], rep(1L, split_sizes[[i]]), axis = -1L)#, function(tsr)
            # tf$squeeze(tsr, axis = 2L)
            # )
        )
      )
      
      res_dist <- do.call(tfd_mixture, args = list(mix, this_components))
      
      return(res_dist)
    }
    
  }
  
  trafo_fun <- function(x){
    
    params <- list()
    for(i in 1:length(trafos_each_param)){
      
      addpart <- 0
      
      if(i>1)
        addpart <- sum(sapply(trafos_each_param[1:(i-1)], 
                              length))
      
      ind <- nr_distributions +
        # first x for pis
        addpart +
        # then for each parameter
        (1:length(trafos_each_param[[i]]))
      
      # print(ind)
      
      for(j in 1:length(trafos_each_param[[i]])){
        
        params <- c(params, 
                    list(trafos_each_param[[i]][[j]](
                      tf_stride_cols(x,ind[j])
                    )))
        
      }
     
      params <- layer_concatenate_identity(params)
       
    }
    
    ret <- c(probs = list(stack(tf$nn$softmax(tf_stride_cols(x, 1L, nr_distributions)),
                                1:nr_distributions)),
             params = params)
    
    if(!is.null(inflation_values))
      ret$params <- layer_concatenate_identity(
        c(ret$params, 
          lapply(inflation_values, function(val) 
            tf$keras$layers$Lambda(function(x) x[,1L:1L])(
              tf$ones_like(ret$params)) * val))
      )
    
    return(ret)
    
  }
  
  return(
    function(x) do.call(mixdist, trafo_fun(x))
  )
  
}


#' generate mixture distribution of same family
#'
#' @param family character; see the \code{family} argument of \code{deepregression}
#' @param nr_comps number of mixture components
#' @param trafos_each_param list of transformations applied before plugging
#' the linear predictor into the parameters of the distributions.
#' Should be of length #parameters of \code{dist}
#' @return returns function than can be used as argument \code{dist\_fun} for
#' @export
#' @rdname dist_maker
#'
samemix_dist_maker <- function(
    family = "normal",
    nr_comps = 3,
    trafos_each_param = NULL
){
  

  return(
    gen_mix_dist_maker(
      type = "same",
      families = family,
      nr_distributions = nr_comps,
      trafos_each_param = trafos_each_param
    )
  )

}
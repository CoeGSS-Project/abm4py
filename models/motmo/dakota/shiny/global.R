##  dakota_restart_util to_tabular dakota.rst dakota.tabular
##  sed -r 's/[ ]+/,/g' ../dakota.tabular > dakota.csv

library(plotly)
library(shiny)
library(tidyverse)
library(magrittr)

## devtools::install_github("rstudio/crosstalk")

options(tibble.width = 120)
options(shiny.autoreload = TRUE)

readTable <- function(filename) { ##//{
    df <- read_table2(filename) %>%
        select(-interface, -scenarioFileName, -calcResultsScript, -X46)
    names(df)[1] <- 'id'
    df
} ##//}

##//{ findVars 
## The response column names starts with a 'o_', so after removing the id column
## and the 'o_*' columns, we have left all the variable names
findVars <- function(df) {
    df %>%
        select(-id) %>%
        names(.) %>%
        Filter(function(s) { ! startsWith(s, 'o_') }, .)
} #//}

##//{ Methods for extracting information from the Response descriptors
  ## In the responce description, a lot of information is encoded, 
  ## namely the variable name, the regiond and the year.
  ## As the description strings starts with "o_", this information
  ## is in the second, third and forth position after a split.
  VARIABLEPOS = 2
  REGIONPOS = 3
  YEARPOS = 4

  findResponses <- function(df) {
    df %>% names(.) %>% Filter(function(s) { startsWith(s, 'o_') }, .)
  }

  extractFromColumnName <- function(cn, pos) {
    cn %>% Map( function(s) { strsplit(s, '_')[[1]][pos] }, . )
  }

  findResponsePart <- function(df, pos) {
    findResponses(df) %>%
    extractFromColumnName(pos) %>%
    unique
  } #//}

createTimeSeriesTable <- function(df) { ##//{
    numR <- length(findResponses(df))
    numT <- length(names(df))
    start <- numT - numR + 1
    df %>%
        gather(key = 'key', value = 'value', start:numT) %>%
        mutate(year = extractFromColumnName(key, YEARPOS) %>% as.integer,
               region = extractFromColumnName(key, REGIONPOS) %>% as.character,
               responseDesc = extractFromColumnName(key, VARIABLEPOS) %>% as.character) %>%
        select(-key)
} ##//}

calcInterval <- function(ts, data, response, lowerFactor, upperFactor) { ##//{
    isInInterval <- function(real, sim, lowerFactor = 0.8, upperFactor = 1.2) {
        ## the == instead of && is intentional, and uses the fact, that
        ## not both bools can be false for a single id. The problem with &&
        ## is, that it accumulate the boolean values of the vector, but we
        ## need a vector 
        (real * lowerFactor < sim) == (sim < real * upperFactor)
    }

    sim <- ts %>% filter(responseDesc == data) %>% select(id, sim = value)
    real <- ts %>% filter(responseDesc == response) %>% select(id, real = value)
    both <- real %>% add_column(sim = sim$sim)
    inInterval <- both %>% group_by(id) %>% summarize(inInterval = sum(isInInterval(sim, real)))

    vecSize <- dim(both)[1] / dim(inInterval)[1]

    vecSize - inInterval$inInterval
} ##//}

calcAbsError <- function(ts, data, response, region) { ##//{
    sim <- ts %>% filter(responseDesc == data) %>% select(id, sim = value)
    real <- ts %>% filter(responseDesc == response) %>% select(id, real = value)
    both <- real %>% add_column(sim = sim$sim)
    (both %>% group_by(id) %>% summarize(error = sqrt(sum((sim - real) ** 2))))$error
} ##//}

createCalibrationTable <- function(df, ts) { ##//{
    numVarsInclId <- length(findVars(df)) + 1

    numDataElecCars <- sum((ts %>% filter(id == 1, responseDesc == 'dataElecCars', region == '6321'))$value)
    numDataCombCars <- sum((ts %>% filter(id == 1, responseDesc == 'dataCombCars', region == '6321'))$value)

    df %>%
        select(1:numVarsInclId) %>%
        mutate(c_intervalElec = calcInterval(ts, 'dataElecCars', 'numElecCars', 0.8, 1.2),
               c_intervalComb = calcInterval(ts, 'dataCombCars', 'numCombCars', 0.8, 1.2)) %>%
        mutate(c_intervalSum = c_intervalElec + c_intervalComb) %>%
        mutate(c_absErrorElec = calcAbsError(ts, 'dataElecCars', 'numElecCars', '6321'),
               c_absErrorComb = calcAbsError(ts, 'dataCombCars', 'numCombCars', '6321')) %>%
        mutate(c_absErrorSum = c_absErrorElec + c_absErrorComb,
               c_absErrorSumWeighted = c_absErrorElec / numDataElecCars + c_absErrorComb / numDataCombCars)
} ##//}

#####------ script style starts here

df <- readTable('../dakota.tabular')

df %<>% mutate_at(vars(contains("dataElec")), funs(. * 3) )
df %<>% mutate_at(vars(contains("dataComb")), funs(. * 0.1) )

ts <- createTimeSeriesTable(df)

ct <- createCalibrationTable(df, ts)

ctl <- ct %>% select(id, starts_with("c_")) %>% gather(key = 'var', value = 'value', -id)



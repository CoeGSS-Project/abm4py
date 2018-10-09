################################################################################ //{ Copyright
##
##    Copyright (c) 2018 Global Climate Forum e.V. 
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################ //}

library(tidyverse)
library(magrittr)

library(shiny)
library(parcoords)

readTable <- function(filename) { ##//{
    df <- read_table2(filename) %>%
        select(-interface, -modelFileName, -calcResponsesScript)
    names(df)[1] <- 'id'
    ## the last column is an artifact from the dakota_restart_util and can be removed
    df %>% select(-length(df))
} ##//}

df <- readTable('../dakota.tabular')


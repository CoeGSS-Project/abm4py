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

library(crosstalk)
library(stringr)

shinyServer(function(input, output, session) {
    cts <- SharedData$new(df %>% select(-runNo),
                          ~id,
                          group = "cts")

    output$all <- renderParcoords({
        parcoords(cts, 
                  brushMode = '1d-axes', reorderable = TRUE, autoresize = TRUE)
    })
    
    cts_params <- SharedData$new(df %>% select(-starts_with('o_'), -runNo),
                                 ~id,
                                 group = "cts")


    
    output$params <- renderParcoords({
        parcoords(cts_params, 
                  brushMode = '1d-axes', reorderable = TRUE, autoresize = TRUE)
    })

    cts_errors <- SharedData$new(df %>% select(id, starts_with('o_'), -runNo),
                                 ~id,
                                 group = "cts")
    
    output$errors <- renderParcoords({
        parcoords(cts_errors, brushMode = '1d-axes', reorderable = TRUE, autoresize = TRUE)
    })
 })

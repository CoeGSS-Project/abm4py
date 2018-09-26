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

asVector <- function(tibble) { unname(unlist(tibble)) }

shinyServer(function(input, output, session) {
    cts <- SharedData$new(ct,
                          ~id,
                          group = "cts")
    tss <- SharedData$new(ts)
    
    doParcoordsPlot <- function(dims) {
        if (input$foo == "None") {
            plot_ly(type = 'parcoords',
                    line = list(color = "blue"),
                    dimensions = dims)        
            
        } else {
            errors <- asVector(ct[input$foo])
            plot_ly(type = 'parcoords', line = list(color = errors,
                                                    colorscale = 'Portland',
                                                    reversescale = TRUE,
                                                    cmin = min(errors),
                                                    cmax = max(errors)),
                    dimensions = dims)
        }
    }


    output$paramParcoords <- renderPlotly({
        cols <- c(names(ct %>% select(-starts_with("c_"))), input$errorCrit)
        dims <- lapply(cols, function(x) { list(label = x, values = asVector(ct[x]))})
        doParcoordsPlot(dims)
    })

    output$errorParcoords <- renderPlotly({
        cols <- c('id', names(ct %>% select(starts_with("c_"))))
        dims <- lapply(cols, function(x) { list(label = x, values = asVector(ct[x]))})
        doParcoordsPlot(dims)
    })


    output$splom <- renderPlotly({
        p <- ct %>%
            plot_ly() %>%
            add_trace(
                type = 'splom',
                dimensions = list(
                    list(label="connRadius",values=~connRadius),
                    list(label="absErrorSum",values=~c_absErrorSum),
                    list(label="mobIncomeShare",values=~mobIncomeShare),
                    list(label="memoryTime", values=~memoryTime)
                )
            )
        p
    })

    
    
    plotTimeSeries <- function(responses) {
        ids <- selectedIds()
        if (length(ids) > 100)
            ids <- ids[1:100]

        print(ids)
        
        ggplot(ts %>% filter(id %in% ids)  %>% filter(responseDesc %in% responses),
               aes(year,
                   value,
                   group = interaction(responseDesc, id),
                   color = responseDesc)) + geom_line()
    }
    
    output$timeSeriesComb <- renderPlotly({
        plotTimeSeries(c('dataCombCars', 'numCombCars'))
    })
    output$timeSeriesElec <- renderPlotly({
        plotTimeSeries(c('dataElecCars', 'numElecCars'))
    })

    cts_params <- SharedData$new(ct %>% select(-starts_with('c_')),
                                 ~id,
                                 group = "cts")

    output$bar <- renderParcoords({
        ## parcoords(cts$data(withSelection = TRUE) %>%
        ##               filter(selected_ | is.na(selected_)) %>%
        ##               select(-selected_),
        parcoords(cts_params, 
                  brushMode = '2d-strums', reorderable = TRUE, autoresize = TRUE)
    })

    cts_errors <- SharedData$new(ct %>% select(id, starts_with('c_')),
                                 ~id,
                                 group = "cts")
    
    output$bar2 <- renderParcoords({
        parcoords(cts_errors, brushMode = '1d-axes', reorderable = TRUE, autoresize = TRUE)
    })
    

    
    output$summary <- renderPrint({
        #TODO : Add warning that only the first n are shown
        cat(length(selectedIds()), "observation(s) selected\n\n")
    })

    selectedIds <- reactive({
        (cts$data(withSelection = TRUE) %>%
          filter(selected_ | is.na(selected_)) %>%
          mutate(selected_ = NULL))$id
    })

    ## Crosstalk demo
    shared_iris <- SharedData$new(iris)

    output$scatter1 <- renderD3scatter({
        d3scatter(shared_iris, ~Petal.Length, ~Petal.Width, ~Species, width = "100%")
    })

    output$scatter2 <- renderD3scatter({
        d3scatter(cts, ~c_absErrorSum, ~c_intervalSum, ~c_absErrorSumWeighted, width = "100%")
        ## d3scatter(shared_iris, ~Sepal.Length, ~Sepal.Width, ~Species, width = "100%")
    })        
})

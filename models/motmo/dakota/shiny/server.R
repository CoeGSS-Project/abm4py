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

asVector <- function(tibble) { unname(unlist(tibble)) }

shinyServer(function(input, output, session) {
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


    output$timeSeries <- renderPlotly({
        p <- ggplot(ts %>% filter(id < 3),
                    aes(year,
                        value,
                        group = interaction(responseDesc, id),
                        color = responseDesc)) + geom_line()
        p
    })
    ## output$timeSeries <- renderPlotly({ 
    ##     ## ts <- yearly

    ##     ## if (input$tsFilterCalibrated) {
    ##     ##     ts <- ts[country %in% gdp_modifier_table$country]
    ##     ## }
    
    ##     ## if (input$tsTopOrManual == 'top10') {
    ##     ##     p <- plotTimeSeries(ts, input$tsRaster, top = input$tsHowMany)
    ##     ## }
    ##     ## if (input$tsTopOrManual == 'top10other') {
    ##     ##     countries <- ts[variable==input$tsTopOfRaster & step==max(ts$step),][order(-value)][1:input$tsHowMany][,country]

    ##     ##     p <- plotTimeSeries(ts, input$tsRaster, countries = countries)
    ##     ## }
    ##     ## if (input$tsTopOrManual == 'manual') {
    ##     ##     p <- plotTimeSeries(ts, input$tsRaster, countries = input$countries)
    ##     ## }
    
    ##     ##ggplotly(p)
    ## })
    ## output$gdpModifier <- DT::renderDataTable(gdp_modifier_table)
})

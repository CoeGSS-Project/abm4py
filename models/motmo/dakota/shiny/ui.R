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

shinyUI( 
    navbarPage(
        "Dakota Visualization",
        ## tabPanel(
        ##     "Crosstalk",
        ##     d3scatterOutput("scatter1"),
        ##     d3scatterOutput("scatter2")
        ## ),
        tabPanel(
            "Time Series",
            plotlyOutput("splom"),
            ## plotlyOutput("scatter3"),
            parcoordsOutput("bar"),
            parcoordsOutput("bar2"),
            verbatimTextOutput("summary"),
            plotlyOutput("timeSeriesComb"),
            plotlyOutput("timeSeriesElec")
        ),
        tabPanel( #//{ TimeSeries
            "Parallal Coordinates",
            fluidRow(
                column(6,
                       selectInput("errorCrit", "Error Criterion", names(ct %>% select(starts_with("c_"))))),
                column(6,
                       selectInput("foo", "Colorscale", c("None", names(ct))))
                ),
            plotlyOutput("paramParcoords"),
            plotlyOutput("errorParcoords")
                    ## plotlyOutput("timeSeries"),
                    ## h3("GDP modifier:"),
                    ## DT::dataTableOutput("gdpModifier")
            ##     )
            ## )
        ) #//}
    )
)

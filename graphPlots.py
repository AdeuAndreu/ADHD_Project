def corrGraphPlot( patientsDF, electrodes, bandPower, numberOfEdges, plotTitle ):
    #Create columns for the band powers
    label_names=[]
    for i in electrodes:
        for j in bandPower:
            combination = i + '_(' + j + ')'
            label_names.append(combination)
        
    node_angles = circular_layout(label_names, label_names, start_pos=90, group_boundaries=[0, len(label_names) / 2 ])
    # Dataframe to plot correlations
    valuesDF = patientsDF[label_names]
    # Create Correlation matrix
    corrDF = valuesDF.corr()
    corrValues = corrDF.values
    # Plot correlationGraph
    plot = plot_connectivity_circle( corrValues, 
                             label_names, 
                             n_lines = numberOfEdges,
                             node_angles=node_angles,
                             colormap  = 'PRGn',
                             facecolor ='White',
                             textcolor = 'Black',
                             title = plotTitle )

######
## HOW TO USE IT: EXAMPLE
##
#patientsDF = loadData( path )
#electrodes = ['Fp1', 'F3', 'C3', 'Fz', 'Cz', 'Fp2', 'F4', 'C4']
#bandPower = ['Theta2+Alpha1', 'Theta', 'Alpha', 'Beta_Global', 'Gamma']
#experiment = 'A'
#numberOfEdges = 100
#plotTitle = 'All-to-All Correlation - Experiment A'
#corrGraphPlot( patientsDF, electrodes, bandPower, numberOfEdges, plotTitle )
##
######
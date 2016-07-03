## Pair plots per electrode and experiment (including bandPower at the end)
#electrodes: List of electrodes to be ploted
#bandPower: List of bands to be ploted
#pathToSavePlot: path where to save pairplots

def pairPlotsPerExperiment ( electrodes, bandPower, pathToSavePlot ):
    bandColumns = [ ]
    bandPowerColumns = [ ]
    for i in electrodes:
        for j in bandPower:
            combinationBand = i + '_(' + j + ')'
            bandColumns.append( combinationBand )
            combinationPower = 'BPR_' + i 
            bandPowerColumns.append( combinationPower )
    numberOfBandPower = len ( bandPower )

    for exp in experiments:
        #For each electrode
        for i in range( 0, len( bandColumns ) - 1, numberOfBandPower ):
            electrodePossition = int( (i + 1 ) / numberOfBandPower )
            electrodeBand =  bandColumns[ i : i + numberOfBandPower ]
            electrodeBand.append( bandPowerColumns[ electrodePossition ] )
            df = patientsDF.loc[ patientsDF.Experiment == exp, electrodeBand ]
            plot = sns.pairplot(df)
            electrode = electrodes[ electrodePossition ]
            picPath = pathToSavePlot + electrode + "_exp_" + exp + ".png"
            plot.savefig( picPath )
            print ("Pair-plot for the electrode " + electrode + " and the experiment " + exp + " has been saved")
            plt.close('all')

######
## HOW TO USE IT: EXAMPLE
##
##electrodes = ['Fp1', 'F3', 'C3', 'Fz', 'Cz', 'Fp2', 'F4', 'C4']
##bandPower = ['Theta', 'Theta2+Alpha1', 'Alpha', 'Beta_Global']
##pathToSavePlot = mainPath + "Plots/"
##pairPlotsPerExperiment ( electrodes, bandPower, pathToSavePlot )
##
######
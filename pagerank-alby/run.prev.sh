#!/bin/bash

BASEDIR="$( dirname "${BASH_SOURCE[0]}" )"
RESULTSDIR=$BASEDIR/../data/results
BINDIR=$BASEDIR
DATE=`date '+%Y_%m_%d_%H_%M_%S'`

echo "Running tests for 'pagerank_experiment' operator..."
echo "vertices, edges, pr_type, num_iterations, final_error, tot_time_ms, preprocessing_time_ms, pr_time_ms" > $RESULTSDIR/results.csv

echo "Computing with full GPU PR"
$BINDIR/pagerank_experiment >> $RESULTSDIR/results_$DATE.csv
echo "Computing with CPU+GPU Sync. PR"
$BINDIR/pagerank_experiment -c >> $RESULTSDIR/results_$DATE.csv
echo "Computing with CPU+GPU Async. PR"
$BINDIR/pagerank_experiment -a >> $RESULTSDIR/results_$DATE.csv

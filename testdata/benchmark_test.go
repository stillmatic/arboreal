package testdata_test

import (
	"encoding/csv"
	"log"
	"os"
	"testing"

	xgb "github.com/Elvenson/xgboost-go"
	"github.com/Elvenson/xgboost-go/activation"
	"github.com/stretchr/testify/assert"
)

func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func BenchmarkXgboostGo(b *testing.B) {
	ensemble, err := xgb.LoadXGBoostFromJSON("mortgage_xgb.json",
		"", 1, 6, &activation.Logistic{})
	assert.NoError(b, err)
	b.Log(ensemble)
	// read data from csv file as struct
	// data, err := mat.ReadCSVFileToDenseMatrix("mortgage_data.csv")
	// assert.NoError(b, err)
	for i := 0; i < b.N; i++ {
		// 	// predict
		// 	// predictions, err := ensemble.PredictProba(data)
		assert.NoError(b, err)
		// 	// _ = predictions
	}

}

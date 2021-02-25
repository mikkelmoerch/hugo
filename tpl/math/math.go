// Copyright 2017 The Hugo Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package math provides template functions for mathmatical operations.
package math

import (
	"errors"
	"math"
	"reflect"

	_math "github.com/gohugoio/hugo/common/math"
	"github.com/spf13/cast"
	"gonum.org/v1/gonum/mat"
)

// New returns a new instance of the math-namespaced template functions.
func New() *Namespace {
	return &Namespace{}
}

// Namespace provides template functions for the "math" namespace.
type Namespace struct{}

// Add adds two numbers.
func (ns *Namespace) Add(a, b interface{}) (interface{}, error) {
	return _math.DoArithmetic(a, b, '+')
}

// Ceil returns the least integer value greater than or equal to x.
func (ns *Namespace) Ceil(x interface{}) (float64, error) {
	xf, err := cast.ToFloat64E(x)
	if err != nil {
		return 0, errors.New("Ceil operator can't be used with non-float value")
	}

	return math.Ceil(xf), nil
}

// Div divides two numbers.
func (ns *Namespace) Div(a, b interface{}) (interface{}, error) {
	return _math.DoArithmetic(a, b, '/')
}

// Floor returns the greatest integer value less than or equal to x.
func (ns *Namespace) Floor(x interface{}) (float64, error) {
	xf, err := cast.ToFloat64E(x)
	if err != nil {
		return 0, errors.New("Floor operator can't be used with non-float value")
	}

	return math.Floor(xf), nil
}

// Log returns the natural logarithm of a number.
func (ns *Namespace) Log(a interface{}) (float64, error) {
	af, err := cast.ToFloat64E(a)

	if err != nil {
		return 0, errors.New("Log operator can't be used with non integer or float value")
	}

	return math.Log(af), nil
}

// Sqrt returns the square root of a number.
// NOTE: will return for NaN for negative values of a
func (ns *Namespace) Sqrt(a interface{}) (float64, error) {
	af, err := cast.ToFloat64E(a)

	if err != nil {
		return 0, errors.New("Sqrt operator can't be used with non integer or float value")
	}

	return math.Sqrt(af), nil
}

// Mod returns a % b.
func (ns *Namespace) Mod(a, b interface{}) (int64, error) {
	ai, erra := cast.ToInt64E(a)
	bi, errb := cast.ToInt64E(b)

	if erra != nil || errb != nil {
		return 0, errors.New("modulo operator can't be used with non integer value")
	}

	if bi == 0 {
		return 0, errors.New("the number can't be divided by zero at modulo operation")
	}

	return ai % bi, nil
}

// ModBool returns the boolean of a % b.  If a % b == 0, return true.
func (ns *Namespace) ModBool(a, b interface{}) (bool, error) {
	res, err := ns.Mod(a, b)
	if err != nil {
		return false, err
	}

	return res == int64(0), nil
}

// Mul multiplies two numbers.
func (ns *Namespace) Mul(a, b interface{}) (interface{}, error) {
	return _math.DoArithmetic(a, b, '*')
}

// Pow returns a raised to the power of b.
func (ns *Namespace) Pow(a, b interface{}) (float64, error) {
	af, erra := cast.ToFloat64E(a)
	bf, errb := cast.ToFloat64E(b)

	if erra != nil || errb != nil {
		return 0, errors.New("Pow operator can't be used with non-float value")
	}

	return math.Pow(af, bf), nil
}

// Round returns the nearest integer, rounding half away from zero.
func (ns *Namespace) Round(x interface{}) (float64, error) {
	xf, err := cast.ToFloat64E(x)
	if err != nil {
		return 0, errors.New("Round operator can't be used with non-float value")
	}

	return _round(xf), nil
}

// Sub subtracts two numbers.
func (ns *Namespace) Sub(a, b interface{}) (interface{}, error) {
	return _math.DoArithmetic(a, b, '-')
}

// MatrixMultiply returns res of matrix multiplication of a value
func (ns *Namespace) MatrixMultiply(v interface{}, cI interface{}, resI interface{}, m []interface{}) (float64, error) {
	// cast interfaces to types
	value, errv := cast.ToFloat64E(v)
	cIndex, errci := cast.ToIntE(cI)
	resIndex, errri := cast.ToIntE(resI)

	if errv != nil {
		return 0, errors.New("MatrixMultiply can't be used with non float value")
	}

	if errci != nil || errri != nil {
		return 0, errors.New("MatrixMultiply can't be used with non integer value")
	}

	// Flatten array of arrays into array
	var d []interface{}
	for _, i := range m {
		c := reflect.ValueOf(i)
		for j := 0; j < c.Len(); j++ {
			d = append(d, c.Index(j).Interface())
		}
	}

	// Create typed float64 array from array d
	mdata := make([]float64, len(d))
	var err error
	for i, unk := range d {
		switch j := unk.(type) {
		case int:
			mdata[i] = float64(j)
		case float64:
			mdata[i] = j
		case float32:
			mdata[i] = float64(j)
		case int64:
			mdata[i] = float64(j)
		// ...other cases...
		default:
			err = errors.New("MatrixMultiply: Unknown value is of incompatible type")
		}
	}

	if err != nil {
		return 0, err
	}

	// Create (dense) matrix from float64 array based on original data
	size := len(m)
	am := mat.NewDense(size, size, mdata)

	// Symmetric mulitplication of value with created matrix
	var resm mat.Dense
	resm.Scale(value, am)

	// Get result at desired index
	return resm.At(resIndex, cIndex), nil
}

// NearFactorize returns the nearest integer, rounding half away from zero.
func (ns *Namespace) NearFactorize(x interface{}) (string, error) {
	xn := cast.ToFloat64(x)

	if xn <= 0 || xn >= 8 {
		return cast.ToString(math.Round(xn)), nil
	}

	wholeNum := ""
	if xn >= 1 {
		flNum := math.Floor(xn)
		wholeNum = cast.ToString(flNum) + " "
		xn = xn - flNum
	}

	if xn == 0 {
		return cast.ToString(x), nil
	}

	diff := float64(1)
	res := ""
	for i := 1; i < 8; i++ {
		for j := 1 + i; j < 9; j++ {
			iFloat := cast.ToFloat64(i)
			jFloat := cast.ToFloat64(j)
			cdiff := math.Abs((iFloat / jFloat) - xn)
			if cdiff < diff {
				res = string(wholeNum + cast.ToString(i) + "/" + cast.ToString(j))
			} else {
				break
			}
			diff = cdiff
		}
	}

	return res, nil
}

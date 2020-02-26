#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../trace.h"
#include "operators.h"

int operator_abs(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output){

	TRACE_LEVEL0("Calling operator_abs\n");
	//Operator accepts exactly one input
	if ( n_input != 1 ) {
		TRACE_LEVEL0("operator_abs has more that one inputs\n");
		return 1;
	}

	//Extract raw info
	if ( input[0]->has_raw_data != NULL ) {
		TRACE_LEVEL0("Input has raw data\n");
		convertRawDataOfTensorProto(input[0]);
	}

	//Set output data type
	if ( input[0]->data_type == 0 ) {
		output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
	} else {
		output[0]->data_type = input[0]->data_type;
	}

	output[0]->has_raw_data = 0;

	output[0]->dims = malloc(input[0]->n_dims * sizeof(output[0]->data_type)); //int64_t
    output[0]->n_dims = input[0]->n_dims;
    output[0]->n_float_data = input[0]->n_float_data; // check other types

    for (int i = 0; i < output[0]->n_dims; i++)
    {
      output[0]->dims[i] = input[0]->dims[i];
    }

	switch(input[0]->data_type)
  	{
	    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		    {
		    	output[0]->n_float_data = input[0]->n_float_data;
		        output[0]->float_data   = malloc(input[0]->n_float_data * sizeof(float));

		    	for (int i = 0; i < output[0]->n_float_data; i++) {
				  	if ( input[0]->float_data[i] < 0 ) {
				  		output[0]->float_data[i] = - input[0]->float_data[i];
				  	} else {
				  		output[0]->float_data[i] = input[0]->float_data[i];
				  	}
		    	}
		    }
		    break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
	        break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
	        break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
	        break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
	        break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
	        break;
	    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
	        break;
	    default:
	        break;
  	}

	return 0;
}
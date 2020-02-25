#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../trace.h"
#include "operators.h"

static int32_t divAndRoundEven(float a, float b)
{
    int32_t quotient  = (int32_t)(a/b); // implicit floor
    int32_t remainder = a - b*quotient;

    if ( remainder > 0 ) {
        quotient++;
    }

    return quotient;
}

// Spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#QuantizeLinear
// y = saturate ((x / y_scale) + y_zero_point)
// output[0] = saturate(input[0]/input[1] + input[2])
int operator_quantizelinear(size_t n_input,
                            Onnx__TensorProto **input,
                            size_t n_attribute,
                            Onnx__AttributeProto **attribute,
                            size_t n_output,
                            Onnx__TensorProto **output)
{
    TRACE_LEVEL0("Calling operator_quantizelinear\n");

    // Initialize "output" tensor
    if ( n_input == 3 ) {
        output[0]->data_type = input[2]->data_type;
    } else {
        // TODO: uint8 [0, 255] or int8 [ -128, 127 ],
        // retrieve 'scale' info according to spec -----> input[1]
        output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UINT8;
    }

    output[0]->has_raw_data = 0;
    output[0]->dims         = malloc(input[0]->n_dims * sizeof(output[0]->data_type));
    output[0]->n_dims       = input[0]->n_dims;

    for (int i = 0; i < output[0]->n_dims; i++)
    {
      output[0]->dims[i] = input[0]->dims[i];
    }

    output[0]->n_int32_data = input[0]->n_float_data;
    output[0]->int32_data   = malloc(output[0]->n_int32_data * sizeof(int32_t));

    //From spec, input is full range (float/int32) thus no need to handle all types
    switch ( input[0]->data_type ) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            {
                for ( int i = 0; i < output[0]->n_int32_data; ++i ) {
                    int32_t value = divAndRoundEven(input[0]->float_data[i], input[1]->float_data[0]);

                    if ( n_input == 3 ) {
                        value += input[2]->int32_data[0];
                    }

                    //TODO saturate either to [0, 255] or [-128, 127]
                    value > 255 ? value = 255 : value;
                    value < 0   ? value = 0   : value;
                    output[0]->int32_data[i] = value;
                }
            }
            break;
            case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
            {
                //TODO complete similarly to FLOAT
            }
            break;
        default:
            printf("Wrong input type!!\n");
            return 1;
            break;
    }

    return 0;
}

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Type definitions
typedef unsigned char uint8; // Define uint8
typedef uint8 image[28][28]; // Define the image type

// Structure definitions
typedef struct LeNet5
{
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];
} LeNet5;

typedef struct Feature
{
    double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    double output[OUTPUT];
} Feature;

// Function declarations
void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);
void Train(LeNet5 *lenet, image input, uint8 label);
uint8 Predict(LeNet5 *lenet, image input, uint8 count);
void Initial(LeNet5 *lenet);

#ifdef __cplusplus
}
#endif

// Network parameters
#define LENGTH_KERNEL    5
#define LENGTH_FEATURE0  32
#define LENGTH_FEATURE1  (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2  (LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3  (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE4  (LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5  (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT            1
#define LAYER1           6
#define LAYER2           6
#define LAYER3           16
#define LAYER4           16
#define LAYER5           120
#define OUTPUT           10

#define INPUT_CHANNELS   1
#define INPUT_SIZE       32
#define LAYER1_CHANNELS  6
#define LAYER1_SIZE      28
#define LAYER2_CHANNELS  16
#define LAYER2_SIZE      14
#define LAYER4_CHANNELS  120
#define LAYER4_SIZE      1
#define KERNEL_SIZE      5

#define ALPHA            0.5
#define PADDING          2
#include <assert.h>
#include <algorithm>
#include <iostream>

#include "ProbPoint.h"

namespace ORB_SLAM2 {

ProbPoint::ProbPoint(int numberOfLabels): mLabel(-1)
{
    // 初始化概率为均匀分布
    vmProb.resize(numberOfLabels, 1.0/numberOfLabels);
}

ProbPoint::ProbPoint(std::vector<float>& vProb) : mLabel(-1)
{
    assert(vProb.size()==3);
    vmProb = vProb;
}

// 递归的贝叶斯概率更新，返回像素点属于的类别
int ProbPoint::BayesianUpdate_old(std::vector<float>& vProb)
{
    if (vProb.size()!=vmProb .size()) {
        cout << "Have some problem!" << endl;
        return -1;
    }

    float maxProb = 0, minProb = 1;
    int label = -1;
    for (size_t i = 0; i < vmProb.size(); i++) {
        vmProb[i] *= vProb[i];
        if (vmProb[i] >= maxProb) {
            maxProb = vmProb[i];
            label = i;
        }else if (vmProb[i] <= minProb) {
            minProb = vmProb[i];
        }
    }
    cout << maxProb << endl;
    float range = maxProb - minProb;
    for (size_t i = 0; i < vmProb.size(); i++) {
        vmProb[i] = (vmProb[i] - minProb) / range;
    }

    mLabel = label;
    return label;
}

int ProbPoint::BayesianUpdate(std::vector<float>& vProb)
{
    if (vProb.size()!=vmProb.size()) {
        cout << "Have some problem!" << endl;
        return -1;
    }

    float maxProb = 0, sum = 0;
    int label = -1;
    for (size_t i = 0; i < vmProb.size(); i++) {
        vmProb[i] *= vProb[i];
        sum += vmProb[i];
        if (vmProb[i] >= maxProb) {
            maxProb = vmProb[i];
            label = i;
        }
    }
    for (size_t i = 0; i < vmProb.size(); i++) {
        vmProb[i] /= sum;
    }

    mLabel = label;
    return label;
}

vector<float>& ProbPoint::getProbVector()
{
    return vmProb;
}

}
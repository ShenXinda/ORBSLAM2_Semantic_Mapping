

#ifndef PROBPOINT_H
#define PROBPOINT_H

#include <vector>
using namespace std;

namespace ORB_SLAM2 {
    
class ProbPoint
{
public:
    ProbPoint(int numberOfLabels);
    ProbPoint(std::vector<float>& vLabels);
    int BayesianUpdate(std::vector<float>& vProb);
    int BayesianUpdate_old(std::vector<float>& vProb);
    vector<float>& getProbVector();

private:
    vector<float> vmProb;  // 属于每个类别的概率
    int mLabel; // 类别标签（0~181）
};

}

#endif
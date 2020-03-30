#pragma once

#include "ITask.h"

template<class T>
class IAsyncTask : public ITask {
public:
    virtual T await() = 0;
};
#pragma once

class ITask {
public:
    bool complete = false;
    virtual void step() = 0;
};
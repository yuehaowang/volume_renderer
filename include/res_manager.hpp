#pragma once
#include <string>

#ifndef RESOUCES_DIRECTORY
#define RESOUCES_DIRECTORY "resources"
#endif


class ResourceManager
{
protected:
    std::string resource_directory;

public:
    ResourceManager();
    std::string getResource(std::string res_name);
};

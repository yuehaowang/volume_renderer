#include "res_manager.hpp"


ResourceManager::ResourceManager()
    : resource_directory(RESOUCES_DIRECTORY)
{
}

std::string ResourceManager::getResource(std::string name)
{
    return resource_directory + "/" + name;
}

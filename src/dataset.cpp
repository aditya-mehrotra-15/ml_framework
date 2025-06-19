#include "ml_framework/dataset.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace ml
{

    static std::string trim(const std::string &s)
    {
        auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos)
            return "";
        auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    Dataset Dataset::load_csv(const std::string &filepath, bool header)
    {
        std::ifstream in(filepath);
        if (!in)
            throw std::runtime_error("Cannot open file " + filepath);
        Dataset D;
        std::string line;
        int rownum = 0;
        if (header && std::getline(in, line))
        {
        } // skip header
        while (std::getline(in, line))
        {
            line = trim(line);
            if (line.empty())
                continue;
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ','))
            {
                cell = trim(cell);
                if (cell.empty())
                    continue;
                try
                {
                    row.push_back(std::stod(cell));
                }
                catch (...)
                {
                    row.clear();
                    break;
                }
            }
            if (row.size() < 2)
                continue;
            D.y.push_back(row.back());
            row.pop_back();
            D.X.push_back(std::move(row));
        }
        return D;
    }

}

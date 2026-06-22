#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class IndexSetList
{
public:
  IndexSetList()
  {
    clear();
  }

  void clear()
  {
    vals_.clear();
    offsets_.resize(1);
    offsets_[0] = 0;
  }

  bool empty() const
  {
    return numSets() == 0;
  }

  Index numSets() const
  {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
  }

  Index numValues() const
  {
    return vals_.size();
  }

  void reserveSets(Index count)
  {
    offsets_.reserve(count + 1);
  }

  void reserveValues(Index count)
  {
    vals_.reserve(count);
  }

  void pushBack(const Vector<Index>& vals)
  {
    if (offsets_.empty())
    {
      offsets_.push_back(0);
    }
    for (Index value : vals)
    {
      vals_.push_back(value);
    }
    offsets_.push_back(vals_.size());
  }

  Index setSize(Index set_id) const
  {
    checkSet(set_id);
    return offsets_[set_id + 1] - offsets_[set_id];
  }

  Vector<Index> set(Index set_id)
  {
    checkSet(set_id);
    const Index begin = offsets_[set_id];
    return Vector<Index>::view(vals_.data() + begin, setSize(set_id));
  }

  Vector<Index> set(Index set_id) const
  {
    checkSet(set_id);
    const Index begin = offsets_[set_id];
    return Vector<Index>::view(
        const_cast<Index*>(vals_.data()) + begin, setSize(set_id));
  }

  Index value(Index set_id,
              Index local_id) const
  {
    checkLocal(set_id, local_id);
    return vals_[offsets_[set_id] + local_id];
  }

  const Index* setData(Index set_id) const
  {
    checkSet(set_id);
    return vals_.data() + offsets_[set_id];
  }

  const Index* offsetsData() const
  {
    return offsets_.data();
  }

  const Index* valuesData() const
  {
    return vals_.data();
  }

private:
  void checkSet(Index set_id) const
  {
    if (set_id < 0 || set_id >= numSets())
    {
      throw std::runtime_error("IndexSetList set is out of range");
    }
  }

  void checkLocal(Index set_id,
                  Index local_id) const
  {
    if (local_id < 0 || local_id >= setSize(set_id))
    {
      throw std::runtime_error("IndexSetList local index is out of range");
    }
  }

private:
  Vector<Index> offsets_;
  Vector<Index> vals_;
};

} // namespace femx

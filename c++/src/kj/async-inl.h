// Copyright (c) 2013, Kenton Varda <temporal@gmail.com>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This file contains extended inline implementation details that are required along with async.h.
// We move this all into a separate file to make async.h more readable.
//
// Non-inline declarations here are defined in async.c++.

#ifndef KJ_ASYNC_H_
#error "Do not include this directly; include kj/async.h."
#endif

#ifndef KJ_ASYNC_INL_H_
#define KJ_ASYNC_INL_H_

namespace kj {
namespace _ {  // private

template <typename T>
class ExceptionOr;

class ExceptionOrValue {
public:
  ExceptionOrValue(bool, Exception&& exception): exception(kj::mv(exception)) {}
  KJ_DISALLOW_COPY(ExceptionOrValue);

  void addException(Exception&& exception) {
    if (this->exception == nullptr) {
      this->exception = kj::mv(exception);
    }
  }

  template <typename T>
  ExceptionOr<T>& as() { return *static_cast<ExceptionOr<T>*>(this); }
  template <typename T>
  const ExceptionOr<T>& as() const { return *static_cast<const ExceptionOr<T>*>(this); }

  Maybe<Exception> exception;

protected:
  // Allow subclasses to have move constructor / assignment.
  ExceptionOrValue() = default;
  ExceptionOrValue(ExceptionOrValue&& other) = default;
  ExceptionOrValue& operator=(ExceptionOrValue&& other) = default;
};

template <typename T>
class ExceptionOr: public ExceptionOrValue {
public:
  ExceptionOr() = default;
  ExceptionOr(T&& value): value(kj::mv(value)) {}
  ExceptionOr(bool, Exception&& exception): ExceptionOrValue(false, kj::mv(exception)) {}
  ExceptionOr(ExceptionOr&&) = default;
  ExceptionOr& operator=(ExceptionOr&&) = default;

  Maybe<T> value;
};

class Event {
  // An event waiting to be executed.  Not for direct use by applications -- promises use this
  // internally.

public:
  Event();
  ~Event() noexcept(false);
  KJ_DISALLOW_COPY(Event);

  void armDepthFirst();
  // Enqueue this event so that `fire()` will be called from the event loop soon.
  //
  // Events scheduled in this way are executed in depth-first order:  if an event callback arms
  // more events, those events are placed at the front of the queue (in the order in which they
  // were armed), so that they run immediately after the first event's callback returns.
  //
  // Depth-first event scheduling is appropriate for events that represent simple continuations
  // of a previous event that should be globbed together for performance.  Depth-first scheduling
  // can lead to starvation, so any long-running task must occasionally yield with
  // `armBreadthFirst()`.  (Promise::then() uses depth-first whereas evalLater() uses
  // breadth-first.)
  //
  // To use breadth-first scheduling instead, use `armLater()`.

  void armBreadthFirst();
  // Like `armDepthFirst()` except that the event is placed at the end of the queue.

  kj::String trace();
  // Dump debug info about this event.

  virtual _::PromiseNode* getInnerForTrace();
  // If this event wraps a PromiseNode, get that node.  Used for debug tracing.
  // Default implementation returns nullptr.

protected:
  virtual Maybe<Own<Event>> fire() = 0;
  // Fire the event.  Possibly returns a pointer to itself, which will be discarded by the
  // caller.  This is the only way that an event can delete itself as a result of firing, as
  // doing so from within fire() will throw an exception.

private:
  friend class kj::EventLoop;
  EventLoop& loop;
  Event* next;
  Event** prev;
  bool firing = false;
};

class PromiseNode {
  // A Promise<T> contains a chain of PromiseNodes tracking the pending transformations.
  //
  // To reduce generated code bloat, PromiseNode is not a template.  Instead, it makes very hacky
  // use of pointers to ExceptionOrValue which actually point to ExceptionOr<T>, but are only
  // so down-cast in the few places that really need to be templated.  Luckily this is all
  // internal implementation details.

public:
  virtual void onReady(Event& event) noexcept = 0;
  // Arms the given event when ready.

  virtual void get(ExceptionOrValue& output) noexcept = 0;
  // Get the result.  `output` points to an ExceptionOr<T> into which the result will be written.
  // Can only be called once, and only after the node is ready.  Must be called directly from the
  // event loop, with no application code on the stack.

  virtual PromiseNode* getInnerForTrace();
  // If this node wraps some other PromiseNode, get the wrapped node.  Used for debug tracing.
  // Default implementation returns nullptr.

protected:
  class OnReadyEvent {
    // Helper class for implementing onReady().

  public:
    void init(Event& newEvent);
    // Returns true if arm() was already called.

    void arm();
    // Arms the event if init() has already been called and makes future calls to init() return
    // true.

  private:
    Event* event = nullptr;
  };
};

// -------------------------------------------------------------------

class ImmediatePromiseNodeBase: public PromiseNode {
public:
  void onReady(Event& event) noexcept override;
};

template <typename T>
class ImmediatePromiseNode final: public ImmediatePromiseNodeBase {
  // A promise that has already been resolved to an immediate value or exception.

public:
  ImmediatePromiseNode(ExceptionOr<T>&& result): result(kj::mv(result)) {}

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

class ImmediateBrokenPromiseNode final: public ImmediatePromiseNodeBase {
public:
  ImmediateBrokenPromiseNode(Exception&& exception);

  void get(ExceptionOrValue& output) noexcept override;

private:
  Exception exception;
};

// -------------------------------------------------------------------

class AttachmentPromiseNodeBase: public PromiseNode {
public:
  AttachmentPromiseNodeBase(Own<PromiseNode>&& dependency);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;

  void dropDependency();

  template <typename>
  friend class AttachmentPromiseNode;
};

template <typename Attachment>
class AttachmentPromiseNode final: public AttachmentPromiseNodeBase {
  // A PromiseNode that holds on to some object (usually, an Own<T>, but could be any movable
  // object) until the promise resolves.

public:
  AttachmentPromiseNode(Own<PromiseNode>&& dependency, Attachment&& attachment)
      : AttachmentPromiseNodeBase(kj::mv(dependency)),
        attachment(kj::mv<Attachment>(attachment)) {}

  ~AttachmentPromiseNode() noexcept(false) {
    // We need to make sure the dependency is deleted before we delete the attachment because the
    // dependency may be using the attachment.
    dropDependency();
  }

private:
  Attachment attachment;
};

// -------------------------------------------------------------------

class TransformPromiseNodeBase: public PromiseNode {
public:
  TransformPromiseNodeBase(Own<PromiseNode>&& dependency);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;

  void dropDependency();
  void getDepResult(ExceptionOrValue& output);

  virtual void getImpl(ExceptionOrValue& output) = 0;

  template <typename, typename, typename, typename>
  friend class TransformPromiseNode;
};

template <typename T, typename DepT, typename Func, typename ErrorFunc>
class TransformPromiseNode final: public TransformPromiseNodeBase {
  // A PromiseNode that transforms the result of another PromiseNode through an application-provided
  // function (implements `then()`).

public:
  TransformPromiseNode(Own<PromiseNode>&& dependency, Func&& func, ErrorFunc&& errorHandler)
      : TransformPromiseNodeBase(kj::mv(dependency)),
        func(kj::fwd<Func>(func)), errorHandler(kj::fwd<ErrorFunc>(errorHandler)) {}

  ~TransformPromiseNode() noexcept(false) {
    // We need to make sure the dependency is deleted before we delete the continuations because it
    // is a common pattern for the continuations to hold ownership of objects that might be in-use
    // by the dependency.
    dropDependency();
  }

private:
  Func func;
  ErrorFunc errorHandler;

  void getImpl(ExceptionOrValue& output) override {
    ExceptionOr<DepT> depResult;
    getDepResult(depResult);
    KJ_IF_MAYBE(depException, depResult.exception) {
      output.as<T>() = handle(
          MaybeVoidCaller<Exception, FixVoid<ReturnType<ErrorFunc, Exception>>>::apply(
              errorHandler, kj::mv(*depException)));
    } else KJ_IF_MAYBE(depValue, depResult.value) {
      output.as<T>() = handle(MaybeVoidCaller<DepT, T>::apply(func, kj::mv(*depValue)));
    }
  }

  ExceptionOr<T> handle(T&& value) {
    return kj::mv(value);
  }
  ExceptionOr<T> handle(PropagateException::Bottom&& value) {
    return ExceptionOr<T>(false, value.asException());
  }
};

// -------------------------------------------------------------------

class ForkHubBase;

class ForkBranchBase: public PromiseNode {
public:
  ForkBranchBase(Own<ForkHubBase>&& hub);
  ~ForkBranchBase() noexcept(false);

  void hubReady() noexcept;
  // Called by the hub to indicate that it is ready.

  // implements PromiseNode ------------------------------------------
  void onReady(Event& event) noexcept override;
  PromiseNode* getInnerForTrace() override;

protected:
  inline ExceptionOrValue& getHubResultRef();

  void releaseHub(ExceptionOrValue& output);
  // Release the hub.  If an exception is thrown, add it to `output`.

private:
  OnReadyEvent onReadyEvent;

  Own<ForkHubBase> hub;
  ForkBranchBase* next = nullptr;
  ForkBranchBase** prevPtr = nullptr;

  friend class ForkHubBase;
};

template <typename T> T copyOrAddRef(T& t) { return t; }
template <typename T> Own<T> copyOrAddRef(Own<T>& t) { return t->addRef(); }

template <typename T>
class ForkBranch final: public ForkBranchBase {
  // A PromiseNode that implements one branch of a fork -- i.e. one of the branches that receives
  // a const reference.

public:
  ForkBranch(Own<ForkHubBase>&& hub): ForkBranchBase(kj::mv(hub)) {}

  void get(ExceptionOrValue& output) noexcept override {
    ExceptionOr<T>& hubResult = getHubResultRef().template as<T>();
    KJ_IF_MAYBE(value, hubResult.value) {
      output.as<T>().value = copyOrAddRef(*value);
    } else {
      output.as<T>().value = nullptr;
    }
    output.exception = hubResult.exception;
    releaseHub(output);
  }
};

// -------------------------------------------------------------------

class ForkHubBase: public Refcounted, protected Event {
public:
  ForkHubBase(Own<PromiseNode>&& inner, ExceptionOrValue& resultRef);

  inline ExceptionOrValue& getResultRef() { return resultRef; }

private:
  Own<PromiseNode> inner;
  ExceptionOrValue& resultRef;

  ForkBranchBase* headBranch = nullptr;
  ForkBranchBase** tailBranch = &headBranch;
  // Tail becomes null once the inner promise is ready and all branches have been notified.

  Maybe<Own<Event>> fire() override;
  _::PromiseNode* getInnerForTrace() override;

  friend class ForkBranchBase;
};

template <typename T>
class ForkHub final: public ForkHubBase {
  // A PromiseNode that implements the hub of a fork.  The first call to Promise::fork() replaces
  // the promise's outer node with a ForkHub, and subsequent calls add branches to that hub (if
  // possible).

public:
  ForkHub(Own<PromiseNode>&& inner): ForkHubBase(kj::mv(inner), result) {}

  Promise<_::UnfixVoid<T>> addBranch() {
    return Promise<_::UnfixVoid<T>>(false, kj::heap<ForkBranch<T>>(addRef(*this)));
  }

private:
  ExceptionOr<T> result;
};

inline ExceptionOrValue& ForkBranchBase::getHubResultRef() {
  return hub->getResultRef();
}

// -------------------------------------------------------------------

class ChainPromiseNode final: public PromiseNode, private Event {
public:
  explicit ChainPromiseNode(Own<PromiseNode> inner);
  ~ChainPromiseNode() noexcept(false);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  enum State {
    STEP1,
    STEP2
  };

  State state;

  Own<PromiseNode> inner;
  // In PRE_STEP1 / STEP1, a PromiseNode for a Promise<T>.
  // In STEP2, a PromiseNode for a T.

  Event* onReadyEvent = nullptr;

  Maybe<Own<Event>> fire() override;
};

template <typename T>
Own<PromiseNode> maybeChain(Own<PromiseNode>&& node, Promise<T>*) {
  return heap<ChainPromiseNode>(kj::mv(node));
}

template <typename T>
Own<PromiseNode>&& maybeChain(Own<PromiseNode>&& node, T*) {
  return kj::mv(node);
}

// -------------------------------------------------------------------

class ExclusiveJoinPromiseNode final: public PromiseNode {
public:
  ExclusiveJoinPromiseNode(Own<PromiseNode> left, Own<PromiseNode> right);
  ~ExclusiveJoinPromiseNode() noexcept(false);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  class Branch: public Event {
  public:
    Branch(ExclusiveJoinPromiseNode& joinNode, Own<PromiseNode> dependency);
    ~Branch() noexcept(false);

    bool get(ExceptionOrValue& output);
    // Returns true if this is the side that finished.

    Maybe<Own<Event>> fire() override;
    _::PromiseNode* getInnerForTrace() override;

  private:
    ExclusiveJoinPromiseNode& joinNode;
    Own<PromiseNode> dependency;
  };

  Branch left;
  Branch right;
  OnReadyEvent onReadyEvent;
};

// -------------------------------------------------------------------

class EagerPromiseNodeBase: public PromiseNode, protected Event {
  // A PromiseNode that eagerly evaluates its dependency even if its dependent does not eagerly
  // evaluate it.

public:
  EagerPromiseNodeBase(Own<PromiseNode>&& dependency, ExceptionOrValue& resultRef);

  void onReady(Event& event) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;
  OnReadyEvent onReadyEvent;

  ExceptionOrValue& resultRef;

  Maybe<Own<Event>> fire() override;
};

template <typename T>
class EagerPromiseNode final: public EagerPromiseNodeBase {
public:
  EagerPromiseNode(Own<PromiseNode>&& dependency)
      : EagerPromiseNodeBase(kj::mv(dependency), result) {}

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

template <typename T>
Own<PromiseNode> spark(Own<PromiseNode>&& node) {
  // Forces evaluation of the given node to begin as soon as possible, even if no one is waiting
  // on it.
  return heap<EagerPromiseNode<T>>(kj::mv(node));
}

// -------------------------------------------------------------------

class AdapterPromiseNodeBase: public PromiseNode {
public:
  void onReady(Event& event) noexcept override;

protected:
  inline void setReady() {
    onReadyEvent.arm();
  }

private:
  OnReadyEvent onReadyEvent;
};

template <typename T, typename Adapter>
class AdapterPromiseNode final: public AdapterPromiseNodeBase,
                                private PromiseFulfiller<UnfixVoid<T>> {
  // A PromiseNode that wraps a PromiseAdapter.

public:
  template <typename... Params>
  AdapterPromiseNode(Params&&... params)
      : adapter(static_cast<PromiseFulfiller<UnfixVoid<T>>&>(*this), kj::fwd<Params>(params)...) {}

  void get(ExceptionOrValue& output) noexcept override {
    KJ_IREQUIRE(!isWaiting());
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
  bool waiting = true;
  Adapter adapter;

  void fulfill(T&& value) override {
    if (waiting) {
      waiting = false;
      result = ExceptionOr<T>(kj::mv(value));
      setReady();
    }
  }

  void reject(Exception&& exception) override {
    if (waiting) {
      waiting = false;
      result = ExceptionOr<T>(false, kj::mv(exception));
      setReady();
    }
  }

  bool isWaiting() override {
    return waiting;
  }
};

}  // namespace _ (private)

// =======================================================================================

template <typename T>
T EventLoop::wait(Promise<T>&& promise) {
  _::ExceptionOr<_::FixVoid<T>> result;

  waitImpl(kj::mv(promise.node), result);

  KJ_IF_MAYBE(value, result.value) {
    KJ_IF_MAYBE(exception, result.exception) {
      throwRecoverableException(kj::mv(*exception));
    }
    return _::returnMaybeVoid(kj::mv(*value));
  } else KJ_IF_MAYBE(exception, result.exception) {
    throwFatalException(kj::mv(*exception));
  } else {
    // Result contained neither a value nor an exception?
    KJ_UNREACHABLE;
  }
}

template <typename Func>
PromiseForResult<Func, void> EventLoop::evalLater(Func&& func) {
  // Invoke thenImpl() on yield().
  return PromiseForResult<Func, void>(false,
      thenImpl(yield(), kj::fwd<Func>(func), _::PropagateException()));
}

template <typename T, typename Func, typename ErrorFunc>
Own<_::PromiseNode> EventLoop::thenImpl(Promise<T>&& promise, Func&& func,
                                        ErrorFunc&& errorHandler) {
  typedef _::FixVoid<_::ReturnType<Func, T>> ResultT;

  Own<_::PromiseNode> intermediate =
      heap<_::TransformPromiseNode<ResultT, _::FixVoid<T>, Func, ErrorFunc>>(
          kj::mv(promise.node), kj::fwd<Func>(func), kj::fwd<ErrorFunc>(errorHandler));
  return _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr));
}

template <typename T>
Promise<T>::Promise(_::FixVoid<T> value)
    : PromiseBase(heap<_::ImmediatePromiseNode<_::FixVoid<T>>>(kj::mv(value))) {}

template <typename T>
Promise<T>::Promise(kj::Exception&& exception)
    : PromiseBase(heap<_::ImmediateBrokenPromiseNode>(kj::mv(exception))) {}

template <typename T>
template <typename Func, typename ErrorFunc>
PromiseForResult<Func, T> Promise<T>::then(Func&& func, ErrorFunc&& errorHandler) {
  return PromiseForResult<Func, T>(false, EventLoop::current().thenImpl(
      kj::mv(*this), kj::fwd<Func>(func), kj::fwd<ErrorFunc>(errorHandler)));
}

template <typename T>
T Promise<T>::wait() {
  return EventLoop::current().wait(kj::mv(*this));
}

template <typename T>
ForkedPromise<T> Promise<T>::fork() {
  return ForkedPromise<T>(false, refcounted<_::ForkHub<_::FixVoid<T>>>(kj::mv(node)));
}

template <typename T>
Promise<T> ForkedPromise<T>::addBranch() {
  return hub->addBranch();
}

template <typename T>
void Promise<T>::exclusiveJoin(Promise<T>&& other) {
  node = heap<_::ExclusiveJoinPromiseNode>(kj::mv(node), kj::mv(other.node));
}

template <typename T>
template <typename... Attachments>
void Promise<T>::attach(Attachments&&... attachments) {
  node = kj::heap<_::AttachmentPromiseNode<Tuple<Attachments...>>>(
      kj::mv(node), kj::tuple(kj::fwd<Attachments>(attachments)...));
}

template <typename T>
void Promise<T>::eagerlyEvaluate() {
  node = _::spark<_::FixVoid<T>>(kj::mv(node));
}

template <typename T>
kj::String Promise<T>::trace() {
  return PromiseBase::trace();
}

template <typename Func>
inline PromiseForResult<Func, void> evalLater(Func&& func) {
  return EventLoop::current().evalLater(kj::fwd<Func>(func));
}

template <typename T>
template <typename ErrorFunc>
void Promise<T>::daemonize(ErrorFunc&& errorHandler) {
  return EventLoop::current().daemonize(then([](T&&) {}, kj::fwd<ErrorFunc>(errorHandler)));
}

template <>
template <typename ErrorFunc>
void Promise<void>::daemonize(ErrorFunc&& errorHandler) {
  return EventLoop::current().daemonize(then([]() {}, kj::fwd<ErrorFunc>(errorHandler)));
}

// =======================================================================================

namespace _ {  // private

template <typename T>
class WeakFulfiller final: public PromiseFulfiller<T>, private kj::Disposer {
  // A wrapper around PromiseFulfiller which can be detached.
  //
  // There are a couple non-trivialities here:
  // - If the WeakFulfiller is discarded, we want the promise it fulfills to be implicitly
  //   rejected.
  // - We cannot destroy the WeakFulfiller until the application has discarded it *and* it has been
  //   detached from the underlying fulfiller, because otherwise the later detach() call will go
  //   to a dangling pointer.  Essentially, WeakFulfiller is reference counted, although the
  //   refcount never goes over 2 and we manually implement the refcounting because we need to do
  //   other special things when each side detaches anyway.  To this end, WeakFulfiller is its own
  //   Disposer -- dispose() is called when the application discards its owned pointer to the
  //   fulfiller and detach() is called when the promise is destroyed.

public:
  KJ_DISALLOW_COPY(WeakFulfiller);

  static kj::Own<WeakFulfiller> make() {
    WeakFulfiller* ptr = new WeakFulfiller;
    return Own<WeakFulfiller>(ptr, *ptr);
  }

  void fulfill(FixVoid<T>&& value) override {
    if (inner != nullptr) {
      inner->fulfill(kj::mv(value));
    }
  }

  void reject(Exception&& exception) override {
    if (inner != nullptr) {
      inner->reject(kj::mv(exception));
    }
  }

  bool isWaiting() override {
    return inner != nullptr && inner->isWaiting();
  }

  void attach(PromiseFulfiller<T>& newInner) {
    inner = &newInner;
  }

  void detach(PromiseFulfiller<T>& from) {
    if (inner == nullptr) {
      // Already disposed.
      delete this;
    } else {
      KJ_IREQUIRE(inner == &from);
      inner = nullptr;
    }
  }

private:
  mutable PromiseFulfiller<T>* inner;

  WeakFulfiller(): inner(nullptr) {}

  void disposeImpl(void* pointer) const override {
    // TODO(perf): Factor some of this out so it isn't regenerated for every fulfiller type?

    if (inner == nullptr) {
      // Already detached.
      delete this;
    } else {
      if (inner->isWaiting()) {
        inner->reject(kj::Exception(
            kj::Exception::Nature::LOCAL_BUG, kj::Exception::Durability::PERMANENT,
            __FILE__, __LINE__,
            kj::heapString("PromiseFulfiller was destroyed without fulfilling the promise.")));
      }
      inner = nullptr;
    }
  }
};

template <typename T>
class PromiseAndFulfillerAdapter {
public:
  PromiseAndFulfillerAdapter(PromiseFulfiller<T>& fulfiller,
                             WeakFulfiller<T>& wrapper)
      : fulfiller(fulfiller), wrapper(wrapper) {
    wrapper.attach(fulfiller);
  }

  ~PromiseAndFulfillerAdapter() noexcept(false) {
    wrapper.detach(fulfiller);
  }

private:
  PromiseFulfiller<T>& fulfiller;
  WeakFulfiller<T>& wrapper;
};

}  // namespace _ (private)

template <typename T, typename Adapter, typename... Params>
Promise<T> newAdaptedPromise(Params&&... adapterConstructorParams) {
  return Promise<T>(false, heap<_::AdapterPromiseNode<_::FixVoid<T>, Adapter>>(
      kj::fwd<Params>(adapterConstructorParams)...));
}

template <typename T>
PromiseFulfillerPair<T> newPromiseAndFulfiller() {
  auto wrapper = _::WeakFulfiller<T>::make();

  Own<_::PromiseNode> intermediate(
      heap<_::AdapterPromiseNode<_::FixVoid<T>, _::PromiseAndFulfillerAdapter<T>>>(*wrapper));
  Promise<_::JoinPromises<T>> promise(false,
      _::maybeChain(kj::mv(intermediate), implicitCast<T*>(nullptr)));

  return PromiseFulfillerPair<T> { kj::mv(promise), kj::mv(wrapper) };
}

}  // namespace kj

#endif  // KJ_ASYNC_INL_H_

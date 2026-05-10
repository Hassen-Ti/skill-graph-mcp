# Frontend Patterns

## Component Design
- Prefer small, single-responsibility components. A component that fits on one screen is easier to test and reason about.
- Co-locate state with the component that owns it. Lift state only when genuinely shared between siblings.
- Use composition over prop-drilling: `children` and render props scale better than passing props 3 levels deep.

## TypeScript Discipline
- Avoid `any`. If you find yourself using it, the type is unknown — model it explicitly.
- Define domain types (e.g. `UserId`, `ISODateString`) instead of using raw primitives everywhere.
- Prefer `type` over `interface` for unions and mapped types; use `interface` for object shapes that will be extended.

## State Management
- Start with `useState` + `useReducer`. Reach for external stores (Zustand, Jotai) only when cross-cutting state becomes painful.
- Server state (fetch, cache, mutations) belongs in a data-fetching layer (React Query, SWR) — do not store it in global client state.
- Keep URL state (filters, pagination) in the query string so pages are bookmarkable and shareable.

## Performance
- Measure before optimizing. Use React DevTools Profiler to identify actual bottlenecks, not hypothetical ones.
- `React.memo` and `useMemo` are not free. Profile first; wrap only when you can see the flame graph benefit.
- Lazy-load heavy routes and components with `React.lazy` + `Suspense`. Keep the initial bundle under 200 kB gzipped.
- Use `IntersectionObserver` for deferred rendering of below-the-fold content.

## Testing
- Test user behaviour, not implementation details. Prefer queries by role/label (Testing Library) over `data-testid`.
- Every interactive component needs at least one keyboard-only test path.
- Visual regression tests (Chromatic, Percy) catch layout regressions that unit tests miss.

## Accessibility
- Every form input needs an associated `<label>`. Use `htmlFor` or wrap with `<label>`.
- Color must not be the only differentiator. Add an icon or text label alongside color-coded status.
- Target touch areas: minimum 44×44 px for interactive elements on mobile.
